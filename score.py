import numpy as np
from geometry import polar2cartesian


def test_policy(policy, env, goal, renderer=None):
    env.reset()
    env.set_goal(goal)
    env.arm.reset()  # force arm to be in vertical configuration
    # import ipdb; ipdb.set_trace()
    obs, rewards, done, info = env.step(env.action_space.sample() * 0)
    while True:
        action, _states = policy.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        if renderer is not None:
            renderer.plot([(env.arm, "tab:blue")])
        if done:
            break
    state = env.arm.get_state()
    pos_ee = env.arm.dynamics.compute_fk(state)
    dist = np.linalg.norm((pos_ee - goal))
    if dist < 0.05:
        return 1.5
    elif dist < 0.1:
        return 1
    else:
        return 0


def score_policy(policy, env, renderer):
    print("\n--- Computing score ---")
    score = 0

    goal = polar2cartesian(1.8, 0.2 - np.pi / 2.0)
    _score = test_policy(policy, env, goal, renderer)
    score += _score
    print(f"\nGoal 1: {_score}")

    goal = polar2cartesian(1.9, -0.15 - np.pi / 2.0)
    _score = test_policy(policy, env, goal, renderer)
    score += _score
    print(f"\nGoal 2: {_score}")

    goal = polar2cartesian(1.6, 0.25 - np.pi / 2.0)
    _score = test_policy(policy, env, goal, renderer)
    score += _score
    print(f"\nGoal 3: {_score}")

    goal = polar2cartesian(1.8, -0.25 - np.pi / 2.0)
    _score = test_policy(policy, env, goal, renderer)
    score += _score
    print(f"\nGoal 4: {_score}")

    goal = polar2cartesian(1.6, 0.45 - np.pi / 2.0)
    _score = test_policy(policy, env, goal, renderer)
    score += _score
    print(f"\nGoal 5: {_score}")

    print('\n\n---')
    print(f'Final score: {score}/7.5')
    print('---')
    return score
