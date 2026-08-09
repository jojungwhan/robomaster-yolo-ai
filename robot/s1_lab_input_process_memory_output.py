"""RoboMaster S1 Lab Python example: one visible systems loop.

Paste this into RoboMaster app > Lab > Python.  RoboMaster Lab supplies
rm_define and the controller objects, so this file is documentation rather than
a normal laptop module.  Test first with the drive wheels raised.
"""


def start():
    robot_ctrl.set_mode(rm_define.robot_mode_free)
    gimbal_ctrl.recenter()
    chassis_ctrl.stop()

    # MEMORY: values survive from one loop pass to the next.
    last_state = "START"
    checks = 0

    while checks < 20:
        # INPUT: the S1 reads whether an armor plate was hit.
        hit = armor_ctrl.check_condition(rm_define.cond_armor_hit)

        # PROCESS: Python turns that input into a named state.
        if hit:
            state = "STOPPED_BY_HIT"
        else:
            state = "OBSERVING"

        # OUTPUT: the robot makes its internal state visible.
        chassis_ctrl.stop()
        if state == "STOPPED_BY_HIT":
            led_ctrl.set_top_led(
                rm_define.armor_top_all,
                255,
                0,
                0,
                rm_define.effect_flash,
            )
            media_ctrl.play_sound(rm_define.media_sound_attacked)
            break
        elif state != last_state:
            led_ctrl.set_top_led(
                rm_define.armor_top_all,
                0,
                0,
                255,
                rm_define.effect_breath,
            )

        # MEMORY + FEEDBACK: save this result, then read the next input.
        last_state = state
        checks = checks + 1
        time.sleep(0.2)

    chassis_ctrl.stop()
