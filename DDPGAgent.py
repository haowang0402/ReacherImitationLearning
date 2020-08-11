from Learner import DDPGLearner
class DDPGAgent():
    def __init__(self, env, env_info, args,hyper_params,learner_cfg, noise_cfg, log_cfg):
        self.curr_state = np.zeros((1,))
        self.total_step = 0
        self.episode_step = 0
        self.i_episode = 0

        self.hyper_params = hyper_params
        self.learner_cfg = learner_cfg
        self.learner_cfg.args = self.args
        self.learner_cfg.env_info = self.env_info
        self.learner_cfg.hyper_params = self.hyper_params
        self.learner_cfg.log_cfg = self.log_cfg
        self.learner_cfg.noise_cfg = noise_cfg
        self.learner_cfg.device = device
        self.noise = OUNoise(
            env_info.action_space.shape[0],
            theta=noise_cfg.ou_noise_theta,
            sigma=noise_cfg.ou_noise_sigma,
        )
        self.learner = DDPGLearner()

    def select_action(self, state):
        """Select an action from the input space."""
        self.curr_state = state
        state = self._preprocess_state(state)
        # if initial random action should be conducted
        if (
            self.total_step < self.hyper_params.initial_random_action
            and not self.args.test
        ):
            return np.array(self.env_info.action_space.sample())

        with torch.no_grad():
            selected_action = self.learner.actor(state).detach().cpu().numpy()

        if not self.args.test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, dict]:
        next_state, reward, done, info = self.env.step(action)

        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = (
                False if self.episode_step == self.args.max_episode_steps else done
            )
            transition = (self.curr_state, action, reward, next_state, done_bool)
            self._add_transition_to_memory(transition)

        return next_state, reward, done, info

    def _add_transition_to_memory(self):
        self.memory.add(transition)

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        i, loss, score, avg_time_cost = log_value
        total_loss = loss.sum()

        print(
            "[INFO] episode %d, episode step: %d, total step: %d, total score: %d\n"
            "total loss: %f actor_loss: %.3f critic_loss: %.3f (spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                total_loss,
                loss[0],
                loss[1],
                avg_time_cost,
            )  # actor loss  # critic loss
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0],
                    "critic loss": loss[1],
                    "time per each step": avg_time_cost,
                }
            )

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            self.set_wandb()

        self.pretrain()

        for self.i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            self.episode_step = 0
            losses = list()

            t_begin = time.time()

            while not done:
                if self.args.render and self.i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                self.total_step += 1
                self.episode_step += 1

                if len(self.memory) >= self.hyper_params.batch_size:
                    for _ in range(self.hyper_params.multiple_update):
                        experience = self.memory.sample()
                        experience = numpy2floattensor(experience)
                        loss = self.learner.update_model(experience)
                        losses.append(loss)  # for logging

                state = next_state
                score += reward

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, score, avg_time_cost)
                self.write_log(log_value)
                losses.clear()

            if self.i_episode % self.args.save_period == 0:
                self.learner.save_params(self.i_episode)
                self.interim_test()

        # termination
        self.env.close()
        self.learner.save_params(self.i_episode)
        self.interim_test()
