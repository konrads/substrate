// This file influenced by code in Substrate's schedule pallet.
// Following is the Apache-2.0 license obligation.

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

//! # SchedulerDatetime tests.

use super::*;
use crate::mock::{logger, new_test_ext, root, run_to_block, Call, LoggerCall, SchedulerDatetime, Test, *};
use frame_support::{
	assert_err, assert_noop, assert_ok,
	traits::{Contains, PreimageProvider},
};
use sp_runtime::traits::Hash;
use chrono_light::prelude::{Frequency};

fn schedule_secs(start: u64, freq: Vec<u32>, end: Option<u64>) -> Schedule {
	let c = Calendar::create();
	Schedule {
		start: c.from_unixtime(start * 1000), // DateTime { year: 1970, month: 1, day: 1, hour: 0, minute: 0, second: start, ms: 0 },
		items: freq.into_iter().map(|x| (Frequency::Second, x)).collect(),
		end: end.map(|x| c.from_unixtime(x * 1000))//DateTime { year: 1970, month: 1, day: 1, hour: 0, minute: 0, second: x, ms: 0 })
	}
}

#[test]
fn basic_scheduling_works() {
	new_test_ext().execute_with(|| {
		let call = Call::Logger(LoggerCall::log { i: 42, weight: 1000 });
		assert!(!<Test as frame_system::Config>::BaseCallFilter::contains(&call));
		assert_ok!(SchedulerDatetime::do_schedule(schedule_secs(20, vec![], None), 127, root(), call.into()));
		run_to_block(3);
		assert!(logger::log().is_empty());
		run_to_block(4);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
		run_to_block(100);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
	});
}

#[test]
fn scheduling_with_preimages_works() {
	new_test_ext().execute_with(|| {
		let call = Call::Logger(LoggerCall::log { i: 42, weight: 1000 });
		let hash = <Test as frame_system::Config>::Hashing::hash_of(&call);
		let hashed = MaybeHashed::Hash(hash.clone());
		assert_ok!(Preimage::note_preimage(Origin::signed(0), call.encode()));
		assert_ok!(SchedulerDatetime::do_schedule(schedule_secs(20, vec![], None), 127, root(), hashed));
		assert!(Preimage::preimage_requested(&hash));
		run_to_block(3);
		assert!(logger::log().is_empty());
		run_to_block(4);
		assert!(!Preimage::have_preimage(&hash));
		assert!(!Preimage::preimage_requested(&hash));
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
		run_to_block(100);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
	});
}

#[test]
fn scheduling_with_preimage_postpones_correctly() {
	new_test_ext().execute_with(|| {
		let call = Call::Logger(LoggerCall::log { i: 42, weight: 1000 });
		let hash = <Test as frame_system::Config>::Hashing::hash_of(&call);
		let hashed = MaybeHashed::Hash(hash.clone());

		assert_ok!(SchedulerDatetime::do_schedule(schedule_secs(20, vec![], None), 127, root(), hashed));
		assert!(Preimage::preimage_requested(&hash));

		run_to_block(4);
		// #4 empty due to no preimage
		assert!(logger::log().is_empty());

		// Register preimage.
		assert_ok!(Preimage::note_preimage(Origin::signed(0), call.encode()));

		run_to_block(5);
		// #5 empty since postponement is 2 blocks.
		assert!(logger::log().is_empty());

		run_to_block(6);
		// #6 is good.
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
		assert!(!Preimage::have_preimage(&hash));
		assert!(!Preimage::preimage_requested(&hash));

		run_to_block(100);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
	});
}

#[test]
fn schedule_after_works() {
	new_test_ext().execute_with(|| {
		run_to_block(2);
		let call = Call::Logger(LoggerCall::log { i: 42, weight: 1000 });
		assert!(!<Test as frame_system::Config>::BaseCallFilter::contains(&call));
		// This will schedule the call 3 blocks after the next block... so block 3 + 3 = 6
		assert_ok!(SchedulerDatetime::do_schedule(schedule_secs(33, vec![], None), 127, root(), call.into()));
		run_to_block(5);
		assert!(logger::log().is_empty());
		run_to_block(6);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
		run_to_block(100);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
	});
}

#[test]
fn schedule_after_zero_works() {
	new_test_ext().execute_with(|| {
		run_to_block(2);
		let call = Call::Logger(LoggerCall::log { i: 42, weight: 1000 });
		assert!(!<Test as frame_system::Config>::BaseCallFilter::contains(&call));
		// Note: deviation from scheduler - do not accept schedules starting *right now*
		assert_noop!(
			SchedulerDatetime::do_schedule(schedule_secs(12, vec![], None), 127, root(), call.clone().into()),
			Error::<Test>::NoFutureScheduleTriggers
		);

		assert_ok!(SchedulerDatetime::do_schedule(schedule_secs(13, vec![], None), 127, root(), call.into()));
		// Will trigger on the next block.
		run_to_block(3);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
		run_to_block(100);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
	});
}

#[test]
fn periodic_scheduling_works() {
	new_test_ext().execute_with(|| {
		// at #4, every 3 blocks, 3 times.
		assert_ok!(SchedulerDatetime::do_schedule(
			schedule_secs(24, vec![18], Some(60)),
			127,
			root(),
			Call::Logger(logger::Call::log { i: 42, weight: 1000 }).into()
		));
		run_to_block(3);
		assert!(logger::log().is_empty());
		run_to_block(4);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
		run_to_block(6);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
		run_to_block(7);
		assert_eq!(logger::log(), vec![(root(), 42u32), (root(), 42u32)]);
		run_to_block(9);
		assert_eq!(logger::log(), vec![(root(), 42u32), (root(), 42u32)]);
		run_to_block(10);
		assert_eq!(logger::log(), vec![(root(), 42u32), (root(), 42u32), (root(), 42u32)]);
		run_to_block(100);
		assert_eq!(logger::log(), vec![(root(), 42u32), (root(), 42u32), (root(), 42u32)]);
	});
}

#[test]
fn reschedule_works() {
	new_test_ext().execute_with(|| {
		let call = Call::Logger(LoggerCall::log { i: 42, weight: 1000 });
		assert!(!<Test as frame_system::Config>::BaseCallFilter::contains(&call));
		assert_eq!(
			SchedulerDatetime::do_schedule(schedule_secs(24, vec![], None), 127, root(), call.into()).unwrap(),
			(4, 0)
		);

		run_to_block(3);
		assert!(logger::log().is_empty());

		assert_eq!(SchedulerDatetime::do_reschedule((4, 0), schedule_secs(36, vec![], None)).unwrap(), (6, 0));

		assert_noop!(
			SchedulerDatetime::do_reschedule((6, 0), schedule_secs(36, vec![], None)),
			Error::<Test>::RescheduleNoChange
		);

		run_to_block(4);
		assert!(logger::log().is_empty());

		run_to_block(6);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);

		run_to_block(100);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
	});
}

#[test]
fn reschedule_named_works() {
	new_test_ext().execute_with(|| {
		let call = Call::Logger(LoggerCall::log { i: 42, weight: 1000 });
		assert!(!<Test as frame_system::Config>::BaseCallFilter::contains(&call));
		assert_eq!(
			SchedulerDatetime::do_schedule_named(
				1u32.encode(),
				schedule_secs(24, vec![], None),
				127,
				root(),
				call.into(),
			)
			.unwrap(),
			(4, 0)
		);

		run_to_block(3);
		assert!(logger::log().is_empty());

		assert_eq!(
			SchedulerDatetime::do_reschedule_named(1u32.encode(), schedule_secs(36, vec![], None)).unwrap(),
			(6, 0)
		);

		assert_noop!(
			SchedulerDatetime::do_reschedule_named(1u32.encode(), schedule_secs(36, vec![], None)),
			Error::<Test>::RescheduleNoChange
		);

		run_to_block(4);
		assert!(logger::log().is_empty());

		run_to_block(6);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);

		run_to_block(100);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
	});
}

#[test]
fn reschedule_named_perodic_works() {
	new_test_ext().execute_with(|| {
		let call = Call::Logger(LoggerCall::log { i: 42, weight: 1000 });
		assert!(!<Test as frame_system::Config>::BaseCallFilter::contains(&call));
		assert_eq!(
			SchedulerDatetime::do_schedule_named(
				1u32.encode(),
				schedule_secs(24, vec![18], Some(60)),
				127,
				root(),
				call.into(),
			)
			.unwrap(),
			(4, 0)
		);

		run_to_block(3);
		assert!(logger::log().is_empty());

		assert_eq!(
			SchedulerDatetime::do_reschedule_named(1u32.encode(), schedule_secs(30, vec![18], Some(66))).unwrap(),
			(5, 0)
		);
		assert_eq!(
			SchedulerDatetime::do_reschedule_named(1u32.encode(), schedule_secs(36, vec![18], Some(72))).unwrap(),
			(6, 0)
		);

		run_to_block(5);
		assert!(logger::log().is_empty());

		run_to_block(6);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);

		assert_eq!(
			SchedulerDatetime::do_reschedule_named(1u32.encode(), schedule_secs(60, vec![18], Some(78))).unwrap(),
			(10, 0)
		);

		run_to_block(9);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);

		run_to_block(10);
		assert_eq!(logger::log(), vec![(root(), 42u32), (root(), 42u32)]);

		run_to_block(13);
		assert_eq!(logger::log(), vec![(root(), 42u32), (root(), 42u32), (root(), 42u32)]);

		run_to_block(100);
		assert_eq!(logger::log(), vec![(root(), 42u32), (root(), 42u32), (root(), 42u32)]);
	});
}

#[test]
fn cancel_named_scheduling_works_with_normal_cancel() {
	new_test_ext().execute_with(|| {
		// at #4.
		SchedulerDatetime::do_schedule_named(
			1u32.encode(),
			schedule_secs(24, vec![], None),
			127,
			root(),
			Call::Logger(LoggerCall::log { i: 69, weight: 1000 }).into(),
		)
		.unwrap();
		let i = SchedulerDatetime::do_schedule(
			schedule_secs(24, vec![], None),
			127,
			root(),
			Call::Logger(LoggerCall::log { i: 42, weight: 1000 }).into(),
		)
		.unwrap();
		run_to_block(3);
		assert!(logger::log().is_empty());
		assert_ok!(SchedulerDatetime::do_cancel_named(None, 1u32.encode()));
		assert_ok!(SchedulerDatetime::do_cancel(None, i));
		run_to_block(100);
		assert!(logger::log().is_empty());
	});
}

#[test]
fn cancel_named_periodic_scheduling_works() {
	new_test_ext().execute_with(|| {
		// at #4, every 3 blocks, 3 times.
		SchedulerDatetime::do_schedule_named(
			1u32.encode(),
			schedule_secs(24, vec![18], Some(60)),
			127,
			root(),
			Call::Logger(LoggerCall::log { i: 42, weight: 1000 }).into(),
		)
		.unwrap();
		// same id results in error.
		assert!(SchedulerDatetime::do_schedule_named(
			1u32.encode(),
			schedule_secs(24, vec![], None),
			127,
			root(),
			Call::Logger(LoggerCall::log { i: 69, weight: 1000 }).into(),
		)
		.is_err());
		// different id is ok.
		SchedulerDatetime::do_schedule_named(
			2u32.encode(),
			schedule_secs(48, vec![], None),
			127,
			root(),
			Call::Logger(LoggerCall::log { i: 69, weight: 1000 }).into(),
		)
		.unwrap();
		run_to_block(3);
		assert!(logger::log().is_empty());
		run_to_block(4);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
		run_to_block(6);
		assert_ok!(SchedulerDatetime::do_cancel_named(None, 1u32.encode()));
		run_to_block(100);
		assert_eq!(logger::log(), vec![(root(), 42u32), (root(), 69u32)]);
	});
}

#[test]
fn scheduler_respects_weight_limits() {
	new_test_ext().execute_with(|| {
		assert_ok!(SchedulerDatetime::do_schedule(
			schedule_secs(24, vec![], None),
			127,
			root(),
			Call::Logger(LoggerCall::log { i: 42, weight: MaximumSchedulerWeight::get() / 2 })
				.into(),
		));
		assert_ok!(SchedulerDatetime::do_schedule(
			schedule_secs(24, vec![], None),
			127,
			root(),
			Call::Logger(LoggerCall::log { i: 69, weight: MaximumSchedulerWeight::get() / 2 })
				.into(),
		));
		// 69 and 42 do not fit together
		run_to_block(4);
		assert_eq!(logger::log(), vec![(root(), 42u32)]);
		run_to_block(5);
		assert_eq!(logger::log(), vec![(root(), 42u32), (root(), 69u32)]);
	});
}

#[test]
fn scheduler_respects_hard_deadlines_more() {
	new_test_ext().execute_with(|| {
		assert_ok!(SchedulerDatetime::do_schedule(
			schedule_secs(24, vec![], None),
			0,
			root(),
			Call::Logger(LoggerCall::log { i: 42, weight: MaximumSchedulerWeight::get() / 2 })
				.into(),
		));
		assert_ok!(SchedulerDatetime::do_schedule(
			schedule_secs(24, vec![], None),
			0,
			root(),
			Call::Logger(LoggerCall::log { i: 69, weight: MaximumSchedulerWeight::get() / 2 })
				.into(),
		));
		// With base weights, 69 and 42 should not fit together, but do because of hard
		// deadlines
		run_to_block(4);
		assert_eq!(logger::log(), vec![(root(), 42u32), (root(), 69u32)]);
	});
}

#[test]
fn scheduler_respects_priority_ordering() {
	new_test_ext().execute_with(|| {
		assert_ok!(SchedulerDatetime::do_schedule(
			schedule_secs(24, vec![], None),
			1,
			root(),
			Call::Logger(LoggerCall::log { i: 42, weight: MaximumSchedulerWeight::get() / 2 })
				.into(),
		));
		assert_ok!(SchedulerDatetime::do_schedule(
			schedule_secs(24, vec![], None),
			0,
			root(),
			Call::Logger(LoggerCall::log { i: 69, weight: MaximumSchedulerWeight::get() / 2 })
				.into(),
		));
		run_to_block(4);
		assert_eq!(logger::log(), vec![(root(), 69u32), (root(), 42u32)]);
	});
}

#[test]
fn scheduler_respects_priority_ordering_with_soft_deadlines() {
	new_test_ext().execute_with(|| {
		let max_weight = MaximumSchedulerWeight::get() - <() as WeightInfo>::on_initialize(0);
		let item_weight =
			<() as WeightInfo>::on_initialize(1) - <() as WeightInfo>::on_initialize(0);
		assert_ok!(SchedulerDatetime::do_schedule(
			schedule_secs(24, vec![], None),
			255,
			root(),
			Call::Logger(LoggerCall::log { i: 42, weight: max_weight / 2 - item_weight }).into(),
		));
		assert_ok!(SchedulerDatetime::do_schedule(
			schedule_secs(24, vec![], None),
			127,
			root(),
			Call::Logger(LoggerCall::log { i: 69, weight: max_weight / 2 - item_weight }).into(),
		));
		assert_ok!(SchedulerDatetime::do_schedule(
			schedule_secs(24, vec![], None),
			126,
			root(),
			Call::Logger(LoggerCall::log { i: 2600, weight: max_weight / 2 - item_weight + 1 })
				.into(),
		));

		// 2600 does not fit with 69 or 42, but has higher priority, so will go through
		run_to_block(4);
		assert_eq!(logger::log(), vec![(root(), 2600u32)]);
		// 69 and 42 fit together
		run_to_block(5);
		assert_eq!(logger::log(), vec![(root(), 2600u32), (root(), 69u32), (root(), 42u32)]);
	});
}

// FIXME: uncomment once weights are fixed!
// #[test]
// fn on_initialize_weight_is_correct() {
// 	new_test_ext().execute_with(|| {
// 		let base_weight = <() as WeightInfo>::on_initialize(0);
// 		let call_weight = MaximumSchedulerWeight::get() / 4;

// 		// Named
// 		assert_ok!(SchedulerDatetime::do_schedule_named(
// 			1u32.encode(),
// 			schedule_secs(18, vec![], None),
// 			255,
// 			root(),
// 			Call::Logger(LoggerCall::log { i: 3, weight: call_weight + 1 }).into(),
// 		));
// 		// Anon Periodic
// 		assert_ok!(SchedulerDatetime::do_schedule(
// 			schedule_secs(12, vec![6000], Some(12012)),
// 			128,
// 			root(),
// 			Call::Logger(LoggerCall::log { i: 42, weight: call_weight + 2 }).into(),
// 		));
// 		// Anon
// 		assert_ok!(SchedulerDatetime::do_schedule(
// 			schedule_secs(12, vec![], None),
// 			127,
// 			root(),
// 			Call::Logger(LoggerCall::log { i: 69, weight: call_weight + 3 }).into(),
// 		));
// 		// Named Periodic
// 		assert_ok!(SchedulerDatetime::do_schedule_named(
// 			2u32.encode(),
// 			schedule_secs(6, vec![6000], Some(12006)),
// 			126,
// 			root(),
// 			Call::Logger(LoggerCall::log { i: 2600, weight: call_weight + 4 }).into(),
// 		));

// 		// Will include the named periodic only
// 		let actual_weight = SchedulerDatetime::on_initialize(1);
// 		assert_eq!(
// 			actual_weight,
// 			base_weight
// 				+ call_weight + 4
// 				+ <() as MarginalWeightInfo>::item(true, true, Some(false))
// 		);
// 		assert_eq!(logger::log(), vec![(root(), 2600u32)]);

// 		// Will include anon and anon periodic
// 		let actual_weight = SchedulerDatetime::on_initialize(2);
// 		assert_eq!(x
// 			actual_weight,
// 			base_weight
// 				+ call_weight + 2
// 				+ <() as MarginalWeightInfo>::item(false, false, Some(false))
// 				+ call_weight + 3
// 				+ <() as MarginalWeightInfo>::item(true, false, Some(false))
// 		);
// 		assert_eq!(logger::log(), vec![(root(), 2600u32), (root(), 69u32), (root(), 42u32)]);

// 		// Will include named only
// 		let actual_weight = SchedulerDatetime::on_initialize(3);
// 		assert_eq!(
// 			actual_weight,
// 			base_weight
// 				+ call_weight + 1
// 				+ <() as MarginalWeightInfo>::item(false, true, Some(false))
// 		);
// 		assert_eq!(
// 			logger::log(),
// 			vec![(root(), 2600u32), (root(), 69u32), (root(), 42u32), (root(), 3u32)]
// 		);

// 		// Will contain none
// 		let actual_weight = SchedulerDatetime::on_initialize(4);
// 		assert_eq!(actual_weight, base_weight);
// 	});
// }

#[test]
fn root_calls_works() {
	new_test_ext().execute_with(|| {
		let call = Box::new(Call::Logger(LoggerCall::log { i: 69, weight: 1000 }).into());
		let call2 = Box::new(Call::Logger(LoggerCall::log { i: 42, weight: 1000 }).into());
		assert_ok!(SchedulerDatetime::schedule_named(Origin::root(), 1u32.encode(), schedule_secs(24, vec![], None), 127, call,));
		assert_ok!(SchedulerDatetime::schedule(Origin::root(), schedule_secs(24, vec![], None), 127, call2));
		run_to_block(3);
		// Scheduled calls are in the agenda.
		assert_eq!(Agenda::<Test>::get(4).len(), 2);
		assert!(logger::log().is_empty());
		assert_ok!(SchedulerDatetime::cancel_named(Origin::root(), 1u32.encode()));
		assert_ok!(SchedulerDatetime::cancel(Origin::root(), 4, 1));
		// Scheduled calls are made NONE, so should not effect state
		run_to_block(100);
		assert!(logger::log().is_empty());
	});
}

#[test]
fn fails_to_schedule_task_in_the_past() {
	new_test_ext().execute_with(|| {
		run_to_block(3);

		let call1 = Box::new(Call::Logger(LoggerCall::log { i: 69, weight: 1000 }).into());
		let call2 = Box::new(Call::Logger(LoggerCall::log { i: 42, weight: 1000 }).into());
		let call3 = Box::new(Call::Logger(LoggerCall::log { i: 42, weight: 1000 }).into());

		assert_err!(
			SchedulerDatetime::schedule_named(Origin::root(), 1u32.encode(), schedule_secs(12, vec![], None), 127, call1),
			Error::<Test>::NoFutureScheduleTriggers,
		);

		assert_err!(
			SchedulerDatetime::schedule(Origin::root(), schedule_secs(12, vec![], None), 127, call2),
			Error::<Test>::NoFutureScheduleTriggers,
		);

		assert_err!(
			SchedulerDatetime::schedule(Origin::root(), schedule_secs(18, vec![], None), 127, call3),
			Error::<Test>::NoFutureScheduleTriggers,
		);
	});
}

#[test]
fn should_use_orign() {
	new_test_ext().execute_with(|| {
		let call = Box::new(Call::Logger(LoggerCall::log { i: 69, weight: 1000 }).into());
		let call2 = Box::new(Call::Logger(LoggerCall::log { i: 42, weight: 1000 }).into());
		assert_ok!(SchedulerDatetime::schedule_named(
			system::RawOrigin::Signed(1).into(),
			1u32.encode(),
			schedule_secs(24, vec![], None),
			127,
			call,
		));
		assert_ok!(SchedulerDatetime::schedule(system::RawOrigin::Signed(1).into(), schedule_secs(24, vec![], None), 127, call2,));
		run_to_block(3);
		// Scheduled calls are in the agenda.
		assert_eq!(Agenda::<Test>::get(4).len(), 2);
		assert!(logger::log().is_empty());
		assert_ok!(SchedulerDatetime::cancel_named(system::RawOrigin::Signed(1).into(), 1u32.encode()));
		assert_ok!(SchedulerDatetime::cancel(system::RawOrigin::Signed(1).into(), 4, 1));
		// Scheduled calls are made NONE, so should not effect state
		run_to_block(100);
		assert!(logger::log().is_empty());
	});
}

#[test]
fn should_check_orign() {
	new_test_ext().execute_with(|| {
		let call = Box::new(Call::Logger(LoggerCall::log { i: 69, weight: 1000 }).into());
		let call2 = Box::new(Call::Logger(LoggerCall::log { i: 42, weight: 1000 }).into());
		assert_noop!(
			SchedulerDatetime::schedule_named(
				system::RawOrigin::Signed(2).into(),
				1u32.encode(),
				schedule_secs(24, vec![], None),
				127,
				call
			),
			BadOrigin
		);
		assert_noop!(
			SchedulerDatetime::schedule(system::RawOrigin::Signed(2).into(), schedule_secs(24, vec![], None), 127, call2),
			BadOrigin
		);
	});
}

#[test]
fn should_check_orign_for_cancel() {
	new_test_ext().execute_with(|| {
		let call =
			Box::new(Call::Logger(LoggerCall::log_without_filter { i: 69, weight: 1000 }).into());
		let call2 =
			Box::new(Call::Logger(LoggerCall::log_without_filter { i: 42, weight: 1000 }).into());
		assert_ok!(SchedulerDatetime::schedule_named(
			system::RawOrigin::Signed(1).into(),
			1u32.encode(),
			schedule_secs(24, vec![], None),
			127,
			call,
		));
		assert_ok!(SchedulerDatetime::schedule(system::RawOrigin::Signed(1).into(), schedule_secs(24, vec![], None), 127, call2,));
		run_to_block(3);
		// Scheduled calls are in the agenda.
		assert_eq!(Agenda::<Test>::get(4).len(), 2);
		assert!(logger::log().is_empty());
		assert_noop!(
			SchedulerDatetime::cancel_named(system::RawOrigin::Signed(2).into(), 1u32.encode()),
			BadOrigin
		);
		assert_noop!(SchedulerDatetime::cancel(system::RawOrigin::Signed(2).into(), 4, 1), BadOrigin);
		assert_noop!(
			SchedulerDatetime::cancel_named(system::RawOrigin::Root.into(), 1u32.encode()),
			BadOrigin
		);
		assert_noop!(SchedulerDatetime::cancel(system::RawOrigin::Root.into(), 4, 1), BadOrigin);
		run_to_block(5);
		assert_eq!(
			logger::log(),
			vec![
				(system::RawOrigin::Signed(1).into(), 69u32),
				(system::RawOrigin::Signed(1).into(), 42u32)
			]
		);
	});
}

#[test]
fn should_fix_clock_drift() {
	// run late @ 10th block
	new_test_ext().execute_with(|| {
		let call = Call::Logger(LoggerCall::log { i: 42, weight: 1000 });
		assert!(!<Test as frame_system::Config>::BaseCallFilter::contains(&call));
		// start at block 5, repeat every 10 blocks
		assert_ok!(SchedulerDatetime::do_schedule(schedule_secs(30, vec![60], None), 127, root(), call.into()));
		run_to_block(4);
		assert!(logger::log().is_empty());
		run_to_block(5);
		assert_eq!(logger::log().len(), 1);

		// inject 12 sec delay, ie. moving schedules 2 blocks forth
		Timestamp::set_timestamp(Timestamp::get() + 12000);
		run_to_block(12);
		assert_eq!(logger::log().len(), 1);
		run_to_block(13);  // instead of 15!
		assert_eq!(logger::log().len(), 2);
		run_to_block(22);
		assert_eq!(logger::log().len(), 2);
		run_to_block(23);
		assert_eq!(logger::log().len(), 3);

		// inject 18 sec speed up, ie. moving schedules 3 blocks back
		Timestamp::set_timestamp(Timestamp::get() - 18000);
		run_to_block(35);
		assert_eq!(logger::log().len(), 3);
		run_to_block(36);  // instead of 33!
		assert_eq!(logger::log().len(), 4);

		// ensure no schedules are lost on delay move back of 60s, 10 blocks
		Timestamp::set_timestamp(Timestamp::get() + 60000);
		run_to_block(39);
		assert_eq!(logger::log().len(), 4);
		run_to_block(40);  // instead of 46!
		assert_eq!(logger::log().len(), 5);
	});
}
