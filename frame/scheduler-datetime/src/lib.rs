// This file is part of Substrate.

// Copyright (C) 2017-2022 Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: Apache-2.0

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Scheduler
//! A Pallet for scheduling dispatches.
//!
//! - [`Config`]
//! - [`Call`]
//! - [`Pallet`]
//!
//! ## Overview
//!
//! This Pallet exposes capabilities for scheduling dispatches to occur at a
//! specified block number or at a specified period. These scheduled dispatches
//! may be named or anonymous and may be canceled.
//!
//! **NOTE:** The scheduled calls will be dispatched with the default filter
//! for the origin: namely `frame_system::Config::BaseCallFilter` for all origin
//! except root which will get no filter. And not the filter contained in origin
//! use to call `fn schedule`.
//!
//! If a call is scheduled using proxy or whatever mecanism which adds filter,
//! then those filter will not be used when dispatching the schedule call.
//!
//! ## Interface
//!
//! ### Dispatchable Functions
//!
//! * `schedule` - schedule a dispatch, which may be periodic, to occur at a specified block and
//!   with a specified priority.
//! * `cancel` - cancel a scheduled dispatch, specified by block number and index.
//! * `schedule_named` - augments the `schedule` interface with an additional `Vec<u8>` parameter
//!   that can be used for identification.
//! * `cancel_named` - the named complement to the cancel function.

// Ensure we're `no_std` when compiling for Wasm.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "runtime-benchmarks")]
mod benchmarking;
#[cfg(test)]
mod mock;
#[cfg(test)]
mod tests;
pub mod weights;

use chrono_light::prelude::{Calendar, Schedule};
use codec::{Codec, Decode, Encode};
use frame_support::{
	dispatch::{DispatchError, DispatchResult, Dispatchable, Parameter},
	traits::{
		schedule_datetime::{self, MaybeHashed},
		EnsureOrigin, Get, IsType, OriginTrait, PrivilegeCmp, StorageVersion, Time,
	},
	weights::{GetDispatchInfo, Weight},
};

use frame_system::{self as system, ensure_signed};
pub use pallet::*;
use scale_info::TypeInfo;
use sp_runtime::{
	traits::{AtLeast32Bit, BadOrigin, One, SaturatedConversion, Saturating},
	RuntimeDebug,
};
use sp_std::{collections::btree_map::BTreeMap, borrow::Borrow, cmp::Ordering, marker::PhantomData, prelude::*};
pub use weights::WeightInfo;

/// Just a simple index for naming period tasks.
pub type PeriodicIndex = u32;
/// The location of a scheduled task that can be used to remove it.
pub type TaskAddress<BlockNumber> = (BlockNumber, u32);

pub type CallOrHashOf<T> = MaybeHashed<<T as Config>::Call, <T as frame_system::Config>::Hash>;

/// Information regarding an item to be executed in the future.
#[cfg_attr(any(feature = "std", test), derive(PartialEq, Eq))]
#[derive(Clone, RuntimeDebug, Encode, Decode, TypeInfo)]
pub struct Scheduled<Call, PalletsOrigin, AccountId> {
	/// The unique identity for this task, if there is one.
	maybe_id: Option<Vec<u8>>,
	/// This task's priority.
	priority: schedule_datetime::Priority,
	/// The call to be dispatched.
	call: Call,
	/// Represents schedule
	schedule: Schedule,
	/// upcoming schedule trigger
	next_trigger_ms: u64,
	/// The origin to dispatch the call.
	origin: PalletsOrigin,
	_phantom: PhantomData<AccountId>,
}

pub type ScheduledOf<T> = Scheduled<
	CallOrHashOf<T>,
	<T as Config>::PalletsOrigin,
	<T as frame_system::Config>::AccountId,
>;

#[cfg(feature = "runtime-benchmarks")]
mod preimage_provider {
	use frame_support::traits::PreimageRecipient;
	pub trait PreimageProviderAndMaybeRecipient<H>: PreimageRecipient<H> {}
	impl<H, T: PreimageRecipient<H>> PreimageProviderAndMaybeRecipient<H> for T {}
}

#[cfg(not(feature = "runtime-benchmarks"))]
mod preimage_provider {
	use frame_support::traits::PreimageProvider;
	pub trait PreimageProviderAndMaybeRecipient<H>: PreimageProvider<H> {}
	impl<H, T: PreimageProvider<H>> PreimageProviderAndMaybeRecipient<H> for T {}
}

pub use preimage_provider::PreimageProviderAndMaybeRecipient;

pub(crate) trait MarginalWeightInfo: WeightInfo {
	fn item(periodic: bool, named: bool, resolved: Option<bool>) -> Weight {
		match (periodic, named, resolved) {
			(_, false, None) => Self::on_initialize_aborted(2) - Self::on_initialize_aborted(1),
			(_, true, None) => {
				Self::on_initialize_named_aborted(2) - Self::on_initialize_named_aborted(1)
			},
			(false, false, Some(false)) => Self::on_initialize(2) - Self::on_initialize(1),
			(false, true, Some(false)) => {
				Self::on_initialize_named(2) - Self::on_initialize_named(1)
			},
			(true, false, Some(false)) => {
				Self::on_initialize_periodic(2) - Self::on_initialize_periodic(1)
			},
			(true, true, Some(false)) => {
				Self::on_initialize_periodic_named(2) - Self::on_initialize_periodic_named(1)
			},
			(false, false, Some(true)) => {
				Self::on_initialize_resolved(2) - Self::on_initialize_resolved(1)
			},
			(false, true, Some(true)) => {
				Self::on_initialize_named_resolved(2) - Self::on_initialize_named_resolved(1)
			},
			(true, false, Some(true)) => {
				Self::on_initialize_periodic_resolved(2) - Self::on_initialize_periodic_resolved(1)
			},
			(true, true, Some(true)) => {
				Self::on_initialize_periodic_named_resolved(2)
					- Self::on_initialize_periodic_named_resolved(1)
			},
		}
	}
}
impl<T: WeightInfo> MarginalWeightInfo for T {}

#[frame_support::pallet]
pub mod pallet {
	use super::*;
	use frame_support::{
		dispatch::PostDispatchInfo,
		pallet_prelude::*,
		traits::{schedule_datetime::LookupError, PreimageProvider},
	};
	use frame_system::pallet_prelude::*;

	/// The current storage version.
	const STORAGE_VERSION: StorageVersion = StorageVersion::new(1);

	#[pallet::pallet]
	// #[pallet::generate_store(pub(super) trait Store)]
	#[pallet::storage_version(STORAGE_VERSION)]
	#[pallet::without_storage_info]
	pub struct Pallet<T>(_);

	/// `system::Config` should always be included in our implied traits.
	#[pallet::config]
	pub trait Config: frame_system::Config {
		type Moment: AtLeast32Bit + Parameter + Default + Copy + MaxEncodedLen;

		/// The overarching event type.
		type Event: From<Event<Self>> + IsType<<Self as frame_system::Config>::Event>;

		/// The aggregated origin which the dispatch will take.
		type Origin: OriginTrait<PalletsOrigin = Self::PalletsOrigin>
			+ From<Self::PalletsOrigin>
			+ IsType<<Self as system::Config>::Origin>;

		/// The caller origin, overarching type of all pallets origins.
		type PalletsOrigin: From<system::RawOrigin<Self::AccountId>> + Codec + Clone + Eq + TypeInfo;

		/// The aggregated call type.
		type Call: Parameter
			+ Dispatchable<Origin = <Self as Config>::Origin, PostInfo = PostDispatchInfo>
			+ GetDispatchInfo
			+ From<system::Call<Self>>;

		/// The maximum weight that may be scheduled per block for any dispatchables of less
		/// priority than `schedule_datetime::HARD_DEADLINE`.
		#[pallet::constant]
		type MaximumWeight: Get<Weight>;

		/// The maximum number of scheduled calls in the queue for a single block.
		/// Not strictly enforced, but used for weight estimation.
		#[pallet::constant]
		type MaxScheduledPerBlock: Get<u32>;

		/// Length of the block, used to calculate schedule wake times.
		#[pallet::constant]
		type ExpectedBlockTime: Get<Self::Moment>;

		/// How often to account for clock drift in schedules.
		#[pallet::constant]
		type ClockDriftFixFrequency: Get<Option<u64>>;

		/// Required origin to schedule or cancel calls.
		type ScheduleOrigin: EnsureOrigin<<Self as system::Config>::Origin>;

		/// Compare the privileges of origins.
		///
		/// This will be used when canceling a task, to ensure that the origin that tries
		/// to cancel has greater or equal privileges as the origin that created the scheduled task.
		///
		/// For simplicity the [`EqualPrivilegeOnly`](frame_support::traits::EqualPrivilegeOnly) can
		/// be used. This will only check if two given origins are equal.
		type OriginPrivilegeCmp: PrivilegeCmp<Self::PalletsOrigin>;

		/// Weight information for extrinsics in this pallet.
		type WeightInfo: WeightInfo;

		/// The preimage provider with which we look up call hashes to get the call.
		type PreimageProvider: PreimageProviderAndMaybeRecipient<Self::Hash>;

		/// If `Some` then the number of blocks to postpone execution for when the item is delayed.
		type NoPreimagePostponement: Get<Option<Self::BlockNumber>>;

		type TimeProvider: Time; // UnixTime;
	}

	/// Items to be executed, indexed by the block number that they should be executed on.
	#[pallet::storage]
	pub type Agenda<T: Config> =
		StorageMap<_, Twox64Concat, T::BlockNumber, Vec<Option<ScheduledOf<T>>>, ValueQuery>;

	/// Lookup from identity to the block number and index of the task.
	#[pallet::storage]
	pub(crate) type Lookup<T: Config> =
		StorageMap<_, Twox64Concat, Vec<u8>, TaskAddress<T::BlockNumber>>;

	/// Events type.
	#[pallet::event]
	#[pallet::generate_deposit(pub(super) fn deposit_event)]
	pub enum Event<T: Config> {
		/// Scheduled some task.
		Scheduled { when: T::BlockNumber, index: u32 },
		/// Canceled some task.
		Canceled { when: T::BlockNumber, index: u32 },
		/// Dispatched some task.
		Dispatched {
			task: TaskAddress<T::BlockNumber>,
			id: Option<Vec<u8>>,
			result: DispatchResult,
		},
		/// The call for the provided hash was not found so the task has been aborted.
		CallLookupFailed {
			task: TaskAddress<T::BlockNumber>,
			id: Option<Vec<u8>>,
			error: LookupError,
		},
	}

	#[pallet::error]
	pub enum Error<T> {
		/// Failed to schedule a call
		FailedToSchedule,
		/// Cannot find the scheduled call.
		NotFound,
		/// Reschedule failed because it does not change scheduled time.
		RescheduleNoChange,
		/// Schedule won't trigger in the future.
		NoFutureScheduleTriggers,
	}

	#[pallet::hooks]
	impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
		/// Execute the scheduled calls
		fn on_initialize(now: T::BlockNumber) -> Weight {
			// FIXME: should take following into the account for weight calculations
			let should_sync_scheduleds = T::ClockDriftFixFrequency::get().map(|x| now.saturated_into::<u64>() % x == 0).unwrap_or_default();
			if should_sync_scheduleds {
				Self::do_sync_scheduleds(now);
			}

			let limit = T::MaximumWeight::get();

			let mut queued = Agenda::<T>::take(now)
				.into_iter()
				.enumerate()
				.filter_map(|(index, s)| Some((index as u32, s?)))
				.collect::<Vec<_>>();

			if queued.len() as u32 > T::MaxScheduledPerBlock::get() {
				log::warn!(
					target: "runtime::scheduler",
					"Warning: This block has more items queued in Scheduler than \
					expected from the runtime configuration. An update might be needed."
				);
			}

			queued.sort_by_key(|(_, s)| s.priority);

			let next = now + One::one();

			let mut total_weight: Weight = <T as Config>::WeightInfo::on_initialize(0);

			if ! queued.is_empty() {
				let now_ms: u64 = T::TimeProvider::now().saturated_into::<u64>();

				for (order, (index, mut s)) in queued.into_iter().enumerate() {
					let named = if let Some(ref id) = s.maybe_id {
						Lookup::<T>::remove(id);
						true
					} else {
						false
					};

					let (call, maybe_completed) = s.call.resolved::<T::PreimageProvider>();
					s.call = call;

					let resolved = if let Some(completed) = maybe_completed {
						T::PreimageProvider::unrequest_preimage(&completed);
						true
					} else {
						false
					};

					let call = match s.call.as_value().cloned() {
						Some(c) => c,
						None => {
							// Preimage not available - postpone until some block.
							total_weight
								.saturating_accrue(<T as Config>::WeightInfo::item(false, named, None));
							if let Some(delay) = T::NoPreimagePostponement::get() {
								let until = now.saturating_add(delay);
								if let Some(ref id) = s.maybe_id {
									let index = Agenda::<T>::decode_len(until).unwrap_or(0);
									Lookup::<T>::insert(id, (until, index as u32));
								}
								Agenda::<T>::append(until, Some(s));
							}
							continue;
						},
					};

					let periodic = !s.schedule.items.is_empty();
					let call_weight = call.get_dispatch_info().weight;
					let mut item_weight =
						<T as Config>::WeightInfo::item(periodic, named, Some(resolved));
					let origin =
						<<T as Config>::Origin as From<T::PalletsOrigin>>::from(s.origin.clone())
							.into();
					if ensure_signed(origin).is_ok() {
						// Weights of Signed dispatches expect their signing account to be whitelisted.
						item_weight.saturating_accrue(T::DbWeight::get().reads_writes(1, 1));
					}

					// We allow a scheduled call if any is true:
					// - It's priority is `HARD_DEADLINE`
					// - It does not push the weight past the limit.
					// - It is the first item in the schedule
					let hard_deadline = s.priority <= schedule_datetime::HARD_DEADLINE;
					let test_weight =
						total_weight.saturating_add(call_weight).saturating_add(item_weight);
					if !hard_deadline && order > 0 && test_weight > limit {
						// Cannot be scheduled this block - postpone until next.
						total_weight
							.saturating_accrue(<T as Config>::WeightInfo::item(false, named, None));
						if let Some(ref id) = s.maybe_id {
							// NOTE: We could reasonably not do this (in which case there would be one
							// block where the named and delayed item could not be referenced by name),
							// but we will do it anyway since it should be mostly free in terms of
							// weight and it is slightly cleaner.
							let index = Agenda::<T>::decode_len(next).unwrap_or(0);
							Lookup::<T>::insert(id, (next, index as u32));
						}
						Agenda::<T>::append(next, Some(s));
						continue;
					}

					let dispatch_origin = s.origin.clone().into();
					let (maybe_actual_call_weight, result) = match call.dispatch(dispatch_origin) {
						Ok(post_info) => (post_info.actual_weight, Ok(())),
						Err(error_and_info) => {
							(error_and_info.post_info.actual_weight, Err(error_and_info.error))
						},
					};
					let actual_call_weight = maybe_actual_call_weight.unwrap_or(call_weight);
					total_weight.saturating_accrue(item_weight);
					total_weight.saturating_accrue(actual_call_weight);

					Self::deposit_event(Event::Dispatched {
						task: (now, index),
						id: s.maybe_id.clone(),
						result,
					});

					// Inject next schedule based on current block number
					if let Some((ms_trigger, block_number_trigger)) = Self::get_next_trigger(&s.schedule, now_ms, now) {
						// If scheduled is named, place its information in `Lookup`
						if let Some(ref id) = s.maybe_id {
							let wake_index = Agenda::<T>::decode_len(block_number_trigger).unwrap_or(0);
							Lookup::<T>::insert(id, (block_number_trigger, wake_index as u32));
						}

						s.next_trigger_ms = ms_trigger;
						Agenda::<T>::append(block_number_trigger, Some(s));
					}
				}
			}
			total_weight
		}
	}

	#[pallet::call]
	impl<T: Config> Pallet<T> {
		#[pallet::weight(<T as Config>::WeightInfo::sync_scheduleds(T::MaxScheduledPerBlock::get()))]
		pub fn sync_scheduleds(origin: OriginFor<T>) -> DispatchResult {
			T::ScheduleOrigin::ensure_origin(origin.clone())?;
			let current_block = <frame_system::Pallet<T>>::block_number();
			Self::do_sync_scheduleds(current_block);
			Ok(())
		}

		/// Anonymously schedule a task.
		#[pallet::weight(<T as Config>::WeightInfo::schedule(T::MaxScheduledPerBlock::get()))]
		pub fn schedule(
			origin: OriginFor<T>,
			schedule: Schedule,
			priority: schedule_datetime::Priority,
			call: Box<CallOrHashOf<T>>,
		) -> DispatchResult {
			T::ScheduleOrigin::ensure_origin(origin.clone())?;
			let origin = <T as Config>::Origin::from(origin);
			Self::do_schedule(
				schedule,
				priority,
				origin.caller().clone(),
				*call,
			)?;
			Ok(())
		}

		/// Cancel an anonymously scheduled task.
		#[pallet::weight(<T as Config>::WeightInfo::cancel(T::MaxScheduledPerBlock::get()))]
		pub fn cancel(origin: OriginFor<T>, when: T::BlockNumber, index: u32) -> DispatchResult {
			T::ScheduleOrigin::ensure_origin(origin.clone())?;
			let origin = <T as Config>::Origin::from(origin);
			Self::do_cancel(Some(origin.caller().clone()), (when, index))?;
			Ok(())
		}

		/// Schedule a named task.
		#[pallet::weight(<T as Config>::WeightInfo::schedule_named(T::MaxScheduledPerBlock::get()))]
		pub fn schedule_named(
			origin: OriginFor<T>,
			id: Vec<u8>,
			schedule: Schedule,
			priority: schedule_datetime::Priority,
			call: Box<CallOrHashOf<T>>,
		) -> DispatchResult {
			T::ScheduleOrigin::ensure_origin(origin.clone())?;
			let origin = <T as Config>::Origin::from(origin);
			Self::do_schedule_named(
				id,
				schedule,
				priority,
				origin.caller().clone(),
				*call,
			)?;
			Ok(())
		}

		/// Cancel a named scheduled task.
		#[pallet::weight(<T as Config>::WeightInfo::cancel_named(T::MaxScheduledPerBlock::get()))]
		pub fn cancel_named(origin: OriginFor<T>, id: Vec<u8>) -> DispatchResult {
			T::ScheduleOrigin::ensure_origin(origin.clone())?;
			let origin = <T as Config>::Origin::from(origin);
			Self::do_cancel_named(Some(origin.caller().clone()), id)?;
			Ok(())
		}

		/// Anonymously schedule a task after a delay.
		///
		/// # <weight>
		/// Same as [`schedule`].
		/// # </weight>
		#[pallet::weight(<T as Config>::WeightInfo::schedule(T::MaxScheduledPerBlock::get()))]
		pub fn schedule_after(
			origin: OriginFor<T>,
			schedule: Schedule,
			priority: schedule_datetime::Priority,
			call: Box<CallOrHashOf<T>>,
		) -> DispatchResult {
			T::ScheduleOrigin::ensure_origin(origin.clone())?;
			let origin = <T as Config>::Origin::from(origin);
			Self::do_schedule(
				schedule,
				priority,
				origin.caller().clone(),
				*call,
			)?;
			Ok(())
		}

		/// Schedule a named task after a delay.
		///
		/// # <weight>
		/// Same as [`schedule_named`](Self::schedule_named).
		/// # </weight>
		#[pallet::weight(<T as Config>::WeightInfo::schedule_named(T::MaxScheduledPerBlock::get()))]
		pub fn schedule_named_after(
			origin: OriginFor<T>,
			id: Vec<u8>,
			schedule: Schedule,
			priority: schedule_datetime::Priority,
			call: Box<CallOrHashOf<T>>,
		) -> DispatchResult {
			T::ScheduleOrigin::ensure_origin(origin.clone())?;
			let origin = <T as Config>::Origin::from(origin);
			Self::do_schedule_named(
				id,
				schedule,
				priority,
				origin.caller().clone(),
				*call,
			)?;
			Ok(())
		}
	}
}

impl<T: Config> Pallet<T> {
	/// (Re)sync schedules:
	/// - recalculates target `wake` BlockNumber (Agenda's key), if clock drift occurred, move it around in Agenda and Lookup storages.
	/// - remove and None in Agenda (presuming not existing in Lookup)
	/// Note: expensive, goes through all of Agenda storage!!!
	fn do_sync_scheduleds(now: T::BlockNumber) {
		let now_ms: u64 = T::TimeProvider::now().saturated_into::<u64>();
		let block_duration: u64 = T::ExpectedBlockTime::get().saturated_into();
		let mut rescheduleds = BTreeMap::<T::BlockNumber, Vec<_>>::new();

		// iterate through all the keys, removing when clock drift detected, keeping track in `rescheduleds` for subsequent re-introduction
		let ignore_error = ();
		for exp_block_number_trigger in Agenda::<T>::iter_keys() {
			Agenda::<T>::try_mutate_exists(exp_block_number_trigger, |scheduled_opts| {
				match scheduled_opts.take() {
					None => Result::Err(ignore_error),  // won't happen, we picked existing key
					Some(scheduleds) => {
						let mut clock_drift_detected = false;
						let indexed_scheduleds = scheduleds.into_iter().filter_map(|x| x);
						let new_scheduled_opts = indexed_scheduleds.filter_map(|x| {
							// detect clock drift
							let block_number_delay = ceil_div(x.next_trigger_ms.saturating_sub(now_ms), block_duration);
							let block_number_trigger = now + block_number_delay.saturated_into();
							if exp_block_number_trigger != block_number_trigger {
								clock_drift_detected = true;
								// remove from Lookup
								if let Some(id) = &x.maybe_id {
									Lookup::<T>::remove(id);
								}

								// store for subsequent insertion
								if let Some(scheduleds) = rescheduleds.get_mut(&block_number_trigger) {
									scheduleds.push(x);
								} else {
									rescheduleds.insert(block_number_trigger, vec![x]);
								}

								None
							} else {
								Some(Some(x))
							}
						}).collect::<Vec<_>>();
						if new_scheduled_opts.is_empty() {
							*scheduled_opts = None;
						} else {
							*scheduled_opts = Some(new_scheduled_opts);
						}

						if clock_drift_detected {
							Ok(())
						} else {
							Result::Err(ignore_error)
						}
					}
				}
			}).ok();
		}

		// re-introduce rescheduled at new BlockNumbers, into both Agenda and Lookup
		for (block_number, scheduleds) in rescheduleds.into_iter() {
			Agenda::<T>::mutate(block_number, |curr_scheduleds| {
				let curr_offset = curr_scheduleds.len();
				for (i, id) in scheduleds.iter().enumerate().filter_map(|(i, x)| Some((i, x.maybe_id.as_ref()?))) {
					Lookup::<T>::insert(&id, (block_number, (i + curr_offset) as u32));
				}
				let mut schedule_opts = scheduleds.into_iter().map(|x|Some(x)).collect::<Vec<_>>();
				curr_scheduleds.append(&mut schedule_opts);
			});
		}
	}

	fn get_next_trigger(schedule: &Schedule, now_ms: u64, curr_block_number: T::BlockNumber) -> Option<(u64, T::BlockNumber)> {
		let calendar = Calendar::create();
		let trigger_in_ms_opt = calendar.next_occurrence_ms(&calendar.from_unixtime(now_ms), schedule);
		trigger_in_ms_opt.map(
			|trigger_in_ms| {
				let ms_trigger = now_ms.saturating_add(trigger_in_ms);
				let block_in_ms = T::ExpectedBlockTime::get().saturated_into();
				let trigger_in_blocks = ceil_div(trigger_in_ms, block_in_ms).max(1);
				let block_number_trigger = curr_block_number.saturating_add(trigger_in_blocks.saturated_into());
				(ms_trigger, block_number_trigger)
			}
		)
	}

	fn do_schedule(
		schedule: Schedule,
		priority: schedule_datetime::Priority,
		origin: T::PalletsOrigin,
		call: CallOrHashOf<T>,
	) -> Result<TaskAddress<T::BlockNumber>, DispatchError> {
		let curr_block_number = <frame_system::Pallet<T>>::block_number();
		let now_ms: u64 = T::TimeProvider::now().saturated_into::<u64>();
		let (next_trigger_ms, next_trigger_block) = Self::get_next_trigger(&schedule, now_ms, curr_block_number).ok_or(Error::<T>::NoFutureScheduleTriggers)?;

		call.ensure_requested::<T::PreimageProvider>();

		let s = Some(Scheduled {
			maybe_id: None,
			priority,
			call,
			schedule,
			next_trigger_ms,
			origin,
			_phantom: PhantomData::<T::AccountId>::default(),
		});
		Agenda::<T>::append(next_trigger_block, s);
		let index = Agenda::<T>::decode_len(next_trigger_block).unwrap_or(1) as u32 - 1;
		Self::deposit_event(Event::Scheduled { when: next_trigger_block, index });

		Ok((next_trigger_block, index))
	}

	fn do_cancel(
		origin: Option<T::PalletsOrigin>,
		(when, index): TaskAddress<T::BlockNumber>,
	) -> Result<(), DispatchError> {
		let scheduled = Agenda::<T>::try_mutate(when, |agenda| {
			agenda.get_mut(index as usize).map_or(
				Ok(None),
				|s| -> Result<Option<Scheduled<_, _, _>>, DispatchError> {
					if let (Some(ref o), Some(ref s)) = (origin, s.borrow()) {
						if matches!(
							T::OriginPrivilegeCmp::cmp_privilege(o, &s.origin),
							Some(Ordering::Less) | None
						) {
							return Err(BadOrigin.into());
						}
					};
					Ok(s.take())
				},
			)
		})?;
		if let Some(s) = scheduled {
			s.call.ensure_unrequested::<T::PreimageProvider>();
			if let Some(id) = s.maybe_id {
				Lookup::<T>::remove(id);
			}
			Self::deposit_event(Event::Canceled { when, index });
			Ok(())
		} else {
			Err(Error::<T>::NotFound)?
		}
	}

	/// Reschedule by (when, index), not sure it makes sense for anonymous schedule
	fn do_reschedule(
		(when, index): TaskAddress<T::BlockNumber>,
		new_schedule: Schedule,
	) -> Result<TaskAddress<T::BlockNumber>, DispatchError> {
		let curr_block_number = <frame_system::Pallet<T>>::block_number();
		let now_ms: u64 = T::TimeProvider::now().saturated_into::<u64>();
		let (_, next_trigger_block) = Self::get_next_trigger(&new_schedule, now_ms, curr_block_number).ok_or(Error::<T>::NoFutureScheduleTriggers)?;

		Agenda::<T>::try_mutate(when, |agenda| -> DispatchResult {
			let task = agenda.get_mut(index as usize).ok_or(Error::<T>::NotFound)?;
			let mut task = task.take().ok_or(Error::<T>::NotFound)?;
			if task.schedule == new_schedule {
				return Err(Error::<T>::RescheduleNoChange.into());
			}
			task.schedule = new_schedule;
			Agenda::<T>::append(next_trigger_block, Some(task));
			Ok(())
		})?;

		let new_index = Agenda::<T>::decode_len(next_trigger_block).unwrap_or(1) as u32 - 1;
		Self::deposit_event(Event::Canceled { when, index });
		Self::deposit_event(Event::Scheduled { when: next_trigger_block, index: new_index });

		Ok((next_trigger_block, new_index))
	}

	fn do_schedule_named(
		id: Vec<u8>,
		schedule: Schedule,
		priority: schedule_datetime::Priority,
		origin: T::PalletsOrigin,
		call: CallOrHashOf<T>,
	) -> Result<TaskAddress<T::BlockNumber>, DispatchError> {
		// ensure id it is unique
		if Lookup::<T>::contains_key(&id) {
			return Err(Error::<T>::FailedToSchedule)?;
		}

		let curr_block_number = <frame_system::Pallet<T>>::block_number();
		let now_ms: u64 = T::TimeProvider::now().saturated_into::<u64>();
		let (next_trigger_ms, next_trigger_block) = Self::get_next_trigger(&schedule, now_ms, curr_block_number).ok_or(Error::<T>::NoFutureScheduleTriggers)?;

		call.ensure_requested::<T::PreimageProvider>();

		let s = Scheduled {
			maybe_id: Some(id.clone()),
			priority,
			call,
			schedule,
			next_trigger_ms,
			origin,
			_phantom: Default::default(),
		};
		Agenda::<T>::append(next_trigger_block, Some(s));
		let index = Agenda::<T>::decode_len(next_trigger_block).unwrap_or(1) as u32 - 1;
		let address = (next_trigger_block, index);
		Lookup::<T>::insert(&id, &address);
		Self::deposit_event(Event::Scheduled { when: next_trigger_block, index });

		Ok(address)
	}

	fn do_cancel_named(origin: Option<T::PalletsOrigin>, id: Vec<u8>) -> DispatchResult {
		Lookup::<T>::try_mutate_exists(id, |lookup| -> DispatchResult {
			if let Some((when, index)) = lookup.take() {
				let i = index as usize;
				Agenda::<T>::try_mutate(when, |agenda| -> DispatchResult {
					if let Some(s) = agenda.get_mut(i) {
						if let (Some(ref o), Some(ref s)) = (origin, s.borrow()) {
							if matches!(
								T::OriginPrivilegeCmp::cmp_privilege(o, &s.origin),
								Some(Ordering::Less) | None
							) {
								return Err(BadOrigin.into());
							}
							s.call.ensure_unrequested::<T::PreimageProvider>();
						}
						*s = None;
					}
					Ok(())
				})?;
				Self::deposit_event(Event::Canceled { when, index });
				Ok(())
			} else {
				Err(Error::<T>::NotFound)?
			}
		})
	}

	fn do_reschedule_named(
		id: Vec<u8>,
		new_schedule: Schedule,
	) -> Result<TaskAddress<T::BlockNumber>, DispatchError> {
		let curr_block_number = <frame_system::Pallet<T>>::block_number();
		let now_ms: u64 = T::TimeProvider::now().saturated_into::<u64>();
		let (_, next_trigger_block) = Self::get_next_trigger(&new_schedule, now_ms, curr_block_number).ok_or(Error::<T>::NoFutureScheduleTriggers)?;

		Lookup::<T>::try_mutate_exists(
			id,
			|lookup| -> Result<TaskAddress<T::BlockNumber>, DispatchError> {
				let (when, index) = lookup.ok_or(Error::<T>::NotFound)?;

				Agenda::<T>::try_mutate(when, |agenda| -> DispatchResult {
					let task = agenda.get_mut(index as usize).ok_or(Error::<T>::NotFound)?;
					let mut task = task.take().ok_or(Error::<T>::NotFound)?;
					if task.schedule == new_schedule {
						return Err(Error::<T>::RescheduleNoChange.into());
					}
					task.schedule = new_schedule;
					Agenda::<T>::append(next_trigger_block, Some(task));

					Ok(())
				})?;

				let new_index = Agenda::<T>::decode_len(next_trigger_block).unwrap_or(1) as u32 - 1;
				Self::deposit_event(Event::Canceled { when, index });
				Self::deposit_event(Event::Scheduled { when: next_trigger_block, index: new_index });

				*lookup = Some((next_trigger_block, new_index));

				Ok((next_trigger_block, new_index))
			},
		)
	}
}

impl<T: Config> schedule_datetime::Anon<T::BlockNumber, <T as Config>::Call, T::PalletsOrigin>
	for Pallet<T>
{
	type Address = TaskAddress<T::BlockNumber>;
	type Hash = T::Hash;

	fn schedule(
		schedule: Schedule,
		priority: schedule_datetime::Priority,
		origin: T::PalletsOrigin,
		call: CallOrHashOf<T>,
	) -> Result<Self::Address, DispatchError> {
		Self::do_schedule(schedule, priority, origin, call)
	}

	fn cancel((when, index): Self::Address) -> Result<(), ()> {
		Self::do_cancel(None, (when, index)).map_err(|_| ())
	}

	fn reschedule(
		address: Self::Address,
		new_schedule: Schedule,
	) -> Result<Self::Address, DispatchError> {
		Self::do_reschedule(address, new_schedule)
	}

	fn next_dispatch_time((when, index): Self::Address) -> Result<T::BlockNumber, ()> {
		Agenda::<T>::get(when).get(index as usize).ok_or(()).map(|_| when)
	}
}

impl<T: Config> schedule_datetime::Named<T::BlockNumber, <T as Config>::Call, T::PalletsOrigin>
	for Pallet<T>
{
	type Address = TaskAddress<T::BlockNumber>;
	type Hash = T::Hash;

	fn schedule_named(
		id: Vec<u8>,
		schedule: Schedule,
		priority: schedule_datetime::Priority,
		origin: T::PalletsOrigin,
		call: CallOrHashOf<T>,
	) -> Result<Self::Address, ()> {
		Self::do_schedule_named(id, schedule, priority, origin, call).map_err(|_| ())
	}

	fn cancel_named(id: Vec<u8>) -> Result<(), ()> {
		Self::do_cancel_named(None, id).map_err(|_| ())
	}

	fn reschedule_named(
		id: Vec<u8>,
		new_schedule: Schedule,
	) -> Result<Self::Address, DispatchError> {
		Self::do_reschedule_named(id, new_schedule)
	}

	fn next_dispatch_time(id: Vec<u8>) -> Result<T::BlockNumber, ()> {
		Lookup::<T>::get(id)
			.and_then(|(when, index)| Agenda::<T>::get(when).get(index as usize).map(|_| when))
			.ok_or(())
	}
}

// utils
fn ceil_div(numerator: u64, denominator: u64) -> u64 {
	numerator.checked_div(denominator).and_then(|x| x.checked_add((numerator % denominator != 0).into())).unwrap()
}

#[cfg(test)]
mod localtests {
    use super::*;

    #[test]
    fn test_ceil_div() {
		assert_eq!(ceil_div(111, 7), 16);
		assert_eq!(ceil_div(112, 7), 16);
		assert_eq!(ceil_div(113, 7), 17);
	}
}
