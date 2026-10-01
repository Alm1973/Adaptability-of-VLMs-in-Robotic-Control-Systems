
import time
from arm import Arm

MOVE_MAGNITUDE = 10


def confirm(prompt):
    answer = input(prompt).strip().lower()
    return answer in ("y", "yes")


def do_move(arm, delta_x, delta_y, label):
    print(f"\n--- {label}: arm.update({delta_x}, {delta_y}) ---")
    base_before, tilt_before = arm.base, arm.tilt

    error = None
    try:
        arm.update(delta_x, delta_y)
    except Exception as e:
        error = e

    base_after, tilt_after = arm.base, arm.tilt
    healthy_after = arm.connection_healthy
    position_changed = (base_after != base_before) or (tilt_after != tilt_before)

    if error is None:
        print(f"Write reported SUCCESS. connection_healthy={healthy_after}")
        print(f"Position tracking: ({base_before},{tilt_before}) -> "
              f"({base_after},{tilt_after})")
        if not position_changed:
            print("  WARNING: write succeeded but tracked position didn't "
                  "change -- unexpected, look into this.")
        return "success", base_before, base_after, tilt_before, tilt_after, healthy_after

    print(f"Write FAILED: {error}")
    print(f"Position tracking: ({base_before},{tilt_before}) -> "
          f"({base_after},{tilt_after}) (should be UNCHANGED if the "
          f"connection-health fix is still working)")
    print(f"connection_healthy is now {healthy_after} (should be False)")
    if position_changed:
        print("  *** FIX NOT WORKING: position advanced despite a failed "
              "write. STOP. ***")
    else:
        print("  Position correctly held steady despite the failed write.")
    return "failed", base_before, base_after, tilt_before, tilt_after, healthy_after


def main():
    print("=== New-firmware verification: 3 moves, human-watched throughout ===\n")
    print("Before anything moves: look at the arm right now.")
    print("arm.py has no way to verify true physical position on its own")
    print("(open-loop servos, no feedback) -- don't trust self.base/self.tilt")
    print("as ground truth, only the physical arm you're watching is.")
    print()
    print("This test will send: base +10, then tilt +10, then base -10.")
    print("Make sure the arm currently has room to move ~10-15 degrees in")
    print("both directions on base, and ~10-15 degrees in the positive")
    print("direction on tilt, before continuing.")
    print()

    if not confirm("Is the arm CURRENTLY in a safe, unstrained position with "
                    "room to move as described above? [yes/no]: "):
        print("Aborted -- no connection opened, no moves sent.")
        return

    print("\nOpening Arm connection...")
    arm = Arm()
    print(f"Arm object's initial position belief (NOT verified, just "
          f"arm.py's hardcoded default): base={arm.base}, tilt={arm.tilt}")
    print("Only the physical arm you're watching is ground truth.")

    results = []

    result = do_move(arm, MOVE_MAGNITUDE, 0, "Move 1/3 (base +10)")
    results.append(("base +10", result))
    if result[0] != "success":
        print("\nStopping -- move 1 did not report success.")
        print_summary(results)
        return

    print("\nWatch the arm now. Waiting 2 seconds...")
    time.sleep(2)
    if not confirm("Did the base joint move as expected (~10 degrees) and "
                    "does everything look fine? [yes/no]: "):
        print("Aborted after move 1 -- human did not confirm it looked correct.")
        print_summary(results)
        return

    result = do_move(arm, 0, MOVE_MAGNITUDE, "Move 2/3 (tilt +10)")
    results.append(("tilt +10", result))
    if result[0] != "success":
        print("\nStopping -- move 2 did not report success.")
        print_summary(results)
        return

    print("\nWatch the arm now. Waiting 2 seconds...")
    time.sleep(2)
    if not confirm("Did the tilt joint move as expected (~10 degrees) and "
                    "does everything look fine? [yes/no]: "):
        print("Aborted after move 2 -- human did not confirm it looked correct.")
        print_summary(results)
        return

    result = do_move(arm, -MOVE_MAGNITUDE, 0, "Move 3/3 (base -10, opposite direction)")
    results.append(("base -10", result))
    if result[0] != "success":
        print("\nStopping -- move 3 did not report success.")
        print_summary(results)
        return

    print("\nWatch the arm now. Waiting 2 seconds...")
    time.sleep(2)
    move3_confirmed = confirm(
        "Did the base joint move back as expected (~10 degrees the other "
        "way) and does everything look fine? [yes/no]: "
    )

    print_summary(results, move3_confirmed_by_human=move3_confirmed)


def print_summary(results, move3_confirmed_by_human=None):
    print("\n=== SUMMARY ===")
    for label, (outcome, base_before, base_after, tilt_before, tilt_after, healthy) in results:
        print(f"{label}: {outcome}, base {base_before}->{base_after}, "
              f"tilt {tilt_before}->{tilt_after}, connection_healthy={healthy}")

    if len(results) < 3:
        print(f"\nSequence stopped early after {len(results)}/3 moves -- "
              f"see reason above. Do not treat this as a passed test.")
        return

    print(f"Move 3 confirmed by human as looking correct: {move3_confirmed_by_human}")

    if all(r[1][0] == "success" for r in results) and move3_confirmed_by_human:
        print("\nAll 3 moves reported success, and move 3 was confirmed by "
              "eye. Combined with your confirmations after moves 1 and 2, "
              "the new firmware's piecewise PWM mapping looks correct for "
              "these small moves in both directions.")
    elif all(r[1][0] == "success" for r in results) and not move3_confirmed_by_human:
        print("\nAll 3 moves reported write SUCCESS, but move 3 was NOT "
              "confirmed as looking correct by eye. Trust the human "
              "observation over the write-success status -- investigate "
              "before further hardware use.")
    else:
        print("\nAt least one move failed. Check above: did position "
              "tracking correctly stay frozen on the failed write(s)? If "
              "so, the connection-health fix is still working even though "
              "something else went wrong -- but this does NOT confirm the "
              "new firmware, since not all 3 moves completed. Investigate "
              "before any further hardware use.")


if __name__ == "__main__":
    main()
