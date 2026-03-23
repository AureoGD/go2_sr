class BaseController:

    def __init__(self):
        pass

    def reset(self):
        pass

    def before_step(self, state, action):
        raise NotImplementedError

    def after_step(self, state):
        pass
