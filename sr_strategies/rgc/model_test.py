import numpy as np
import pinocchio as pin
from sr_strategies.tsm.smooth_filter import SmoothFilter
from sr_strategies.rgc.roll_cw import RollCW
from sr_strategies.rgc.stand_up import StandUpPhase


class ControlScheduller():

    def __init__(self, **kwargs):
        self.robot_states = kwargs.get('robot_states', [])
        # configure "start_pos" smooth filer
        qr = np.array([[0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7]]).transpose()
        go_safe_kwargd = {'robot_states': self.robot_states, 'settling_time': 1, 't_cont': 0.01, 'qHL': qr}
        self.go_safe = SmoothFilter(**go_safe_kwargd)

        qr = np.array([[0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.3, -2.7, -1.1, 1.4, -2.8]]).transpose()
        prepare_to_rool1 = {'robot_states': self.robot_states, 'settling_time': 1.5, 't_cont': 0.01, 'qHL': qr}
        self.prepare_to_rool1 = SmoothFilter(**prepare_to_rool1)

        qr = np.array([[0, 1.4, -2.7, -0.50, 1.0, -2.7, 0, 1.3, -2.7, -0.75, 4, -2.8]]).transpose()
        prepare_to_rool2 = {'robot_states': self.robot_states, 'settling_time': 1.5, 't_cont': 0.01, 'qHL': qr}
        self.prepare_to_rool2 = SmoothFilter(**prepare_to_rool2)

        # -0.6, 0, -2.8, -0.8, 1.0, -2.8, -0.6, 0, -2.8, -0.5, 4.45, -2.5
        # -0.75, 1.4, -2.7, -0.50, 1.0, -2.7, -0.6, 1.3, -2.7, -0.5, 4.3, -2.8
        qr = np.array([[-0.6, 0, -2.8, -0.8, 1.0, -2.8, -0.6, 0, -2.8, -0.5, 4.45, -2.5]]).transpose()

        shoulder_touch = {'robot_states': self.robot_states, 'settling_time': 0.5, 't_cont': 0.01, 'qHL': qr}
        self.shoulder_touch = SmoothFilter(**shoulder_touch)

        qr = np.array([[-0.75, 1.4, -2.7, -0.50, 1.0, -2.7, -0.6, 1.3, -2.7, -0.0, 3.3, -1.3]]).transpose()
        push_to_roll = {'robot_states': self.robot_states, 'settling_time': 10, 't_cont': 0.01, 'qHL': qr}
        self.push_to_roll = SmoothFilter(**push_to_roll)

        # qr = np.array([[0.25, 0.90, -2.85, -0.85, 0.85, -1.3, 0.25, 0.90, -2.85, -0.9, 0.9, -1.5]]).transpose()
        qr = np.array([[-0.25, 1.5, -2.0, -0.85, 0.85, -1.3, -0.25, 1.5, -2.2, 0.6, 3.75, -1.5]]).transpose()
        landing_stage_1 = {'robot_states': self.robot_states, 'settling_time': 3, 't_cont': 0.01, 'qHL': qr}
        self.landing_stage_1 = SmoothFilter(**landing_stage_1)

        qr = np.array([[-0.25, 1.5, -2.0, -0.85, 0.85, -1.3, -0.25, 1.5, -2.2, -0.9, 0.9, -1.5]]).transpose()
        landing_stage_2 = {'robot_states': self.robot_states, 'settling_time': 2, 't_cont': 0.01, 'qHL': qr}
        self.landing_stage_2 = SmoothFilter(**landing_stage_2)

        qr = np.array([[0.5, 1.5, -2.0, -0.85, 0.85, -2.0, 0.5, 1.5, -2.2, -0.9, 0.9, -2.0]]).transpose()
        landing_stage_3 = {'robot_states': self.robot_states, 'settling_time': 0.5, 't_cont': 0.01, 'qHL': qr}
        self.landing_stage_3 = SmoothFilter(**landing_stage_3)

        qr = np.array([[0.5, 1.5, -2.7, -0.85, 0.5, -1.7, 0.5, 1.5, -2.7, -0.9, 0.5, -1.7]]).transpose()
        landing_stage_4 = {'robot_states': self.robot_states, 'settling_time': 0.5, 't_cont': 0.01, 'qHL': qr}
        self.landing_stage_4 = SmoothFilter(**landing_stage_4)

        qr = np.array([[1.05, 1.5, -2.7, -0.85, 1.4, -2.7, 1.05, 1.5, -2.7, -0.9, 1.4, -2.7]]).transpose()
        landing_stage_5 = {'robot_states': self.robot_states, 'settling_time': 0.5, 't_cont': 0.01, 'qHL': qr}
        self.landing_stage_5 = SmoothFilter(**landing_stage_5)

        qr = np.array([[1.05, 1.5, -2.7, -0.85, 1.4, -2.7, 0, 1.55, -2.7, -0.9, 1.4, -2.7]]).transpose()
        landing_stage_6 = {'robot_states': self.robot_states, 'settling_time': 0.5, 't_cont': 0.01, 'qHL': qr}
        self.landing_stage_6 = SmoothFilter(**landing_stage_6)

        qr = np.array([[0, 1.5, -2.7, -0.85, 1.4, -2.7, 0, 1.55, -2.7, -0.9, 1.4, -2.7]]).transpose()
        landing_stage_7 = {'robot_states': self.robot_states, 'settling_time': 0.5, 't_cont': 0.01, 'qHL': qr}
        self.landing_stage_7 = SmoothFilter(**landing_stage_7)

        qr = np.array([[-0.2, 1.0, -2.5, 0.2, 1.0, -2.5, -0.2, 1.0, -2.5, 0.2, 1.0, -2.5]]).transpose()
        # qr = np.array([[0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7, 0, 1.4, -2.7]]).transpose()
        landing_stage_8 = {'robot_states': self.robot_states, 'settling_time': 0.5, 't_cont': 0.01, 'qHL': qr}
        self.landing_stage_8 = SmoothFilter(**landing_stage_8)

        self.roll_clockwise = RollCW(**kwargs)
        self.stand_up = StandUpPhase(**kwargs)

        self.kp = kwargs.get('kp')
        self.kd = kwargs.get('kd')

        self.KP = self.kp * np.eye(12)
        self.KD = self.kd * np.eye(12)

        self.using_mpc = False

        self.tick = 0

    def update(self, mode):
        if self.tick * 0.01 <= 1.5:
            delta_qr = self.go_safe.smooth_reference().reshape(12, 1)
        elif self.tick * 0.01 > 1.5 and self.tick * 0.01 <= 3:
            delta_qr = self.prepare_to_rool1.smooth_reference().reshape(12, 1)
        elif self.tick * 0.01 > 3 and self.tick * 0.01 <= 4:
            delta_qr = self.prepare_to_rool2.smooth_reference().reshape(12, 1)
        elif self.tick * 0.01 > 4 and self.tick * 0.01 <= 4.75:
            delta_qr = self.shoulder_touch.smooth_reference().reshape(12, 1)
        elif self.tick * 0.01 > 4.75 and self.tick * 0.01 <= 9:
            self.using_mpc = True
            delta_qr = self.roll_clockwise.solve_rgc().reshape(12, 1)
            self.KD[0:3, 0:3] = self.kd / 10 * np.eye(3)
        elif self.tick * 0.01 > 9 and self.tick * 0.01 <= 12:
            self.KD[0:3, 0:3] = self.kd * np.eye(3)
            delta_qr = self.landing_stage_1.smooth_reference().reshape(12, 1)
        elif self.tick * 0.01 > 12 and self.tick * 0.01 <= 12.5:
            delta_qr = self.landing_stage_2.smooth_reference().reshape(12, 1)
        elif self.tick * 0.01 > 12.5 and self.tick * 0.01 <= 13:
            delta_qr = self.landing_stage_3.smooth_reference().reshape(12, 1)
        elif self.tick * 0.01 > 13 and self.tick * 0.01 <= 13.5:
            delta_qr = self.landing_stage_4.smooth_reference().reshape(12, 1)
        elif self.tick * 0.01 > 13.5 and self.tick * 0.01 <= 14:
            delta_qr = self.landing_stage_5.smooth_reference().reshape(12, 1)
        elif self.tick * 0.01 > 14 and self.tick * 0.01 <= 14.5:
            delta_qr = self.landing_stage_6.smooth_reference().reshape(12, 1)
        elif self.tick * 0.01 > 14.5 and self.tick * 0.01 <= 15:
            delta_qr = self.landing_stage_7.smooth_reference().reshape(12, 1)
        elif self.tick * 0.01 > 15 and self.tick * 0.01 <= 15.5:
            delta_qr = self.landing_stage_8.smooth_reference().reshape(12, 1)
        else:
            delta_qr = self.stand_up.solve_rgc().reshape(12, 1)

        self.tick += 1
        qr = self.robot_states.qr + delta_qr
        return qr.reshape(12), self.KP, self.KD

        # if self.tick * 0.01 <= 3:
        #     delta_qr = self.landing_stage_1.smooth_reference().reshape(12, 1)
        # elif self.tick * 0.01 > 3 and self.tick * 0.01 <= 3.5:
        #     delta_qr = self.landing_stage_2.smooth_reference().reshape(12, 1)
        # elif self.tick * 0.01 > 3.5 and self.tick * 0.01 <= 4:
        #     delta_qr = self.landing_stage_3.smooth_reference().reshape(12, 1)
        # elif self.tick * 0.01 > 4 and self.tick * 0.01 <= 4.5:
        #     delta_qr = self.landing_stage_4.smooth_reference().reshape(12, 1)
        # elif self.tick * 0.01 > 4.5 and self.tick * 0.01 <= 5:
        #     delta_qr = self.landing_stage_5.smooth_reference().reshape(12, 1)
        # elif self.tick * 0.01 > 5 and self.tick * 0.01 <= 5.5:
        #     delta_qr = self.landing_stage_6.smooth_reference().reshape(12, 1)
        # elif self.tick * 0.01 > 5.5 and self.tick * 0.01 <= 6:
        #     delta_qr = self.landing_stage_7.smooth_reference().reshape(12, 1)
        # elif self.tick * 0.01 > 6 and self.tick * 0.01 <= 6.5:
        #     delta_qr = self.landing_stage_8.smooth_reference().reshape(12, 1)
        # else:
        #     delta_qr = self.stand_up.solve_rgc().reshape(12, 1)
