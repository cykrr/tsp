class ModelEvalActions():
  def __init__(self, model):
    self.model=model

  # permite evaluar acctiones de varios estados a la vez
  # para optimizar los cáluclos del modelo
  def __call__(self, states, env):
    single_state = False
    if not isinstance(states, list):
      single_state=True
      states = [states]

    evals = [list() for _ in states]
    vecSeqs=[]; move2idx =[]

    for state in states:
      vecSeq, _, mov2idx = state.state2vecSeq()
      vecSeqs.append(vecSeq)
      move2idx.append(mov2idx)

    predictions = self.model(torch.tensor(vecSeqs), return_probabilities=True)

    for k in range(len(states)):
      state = states[k]
      for action in env.gen_actions(state, "constructive"):
          idx = move2idx[k][action[1]] #mapping from move to output index (model)
          evals[k].append((action,predictions[k][idx]))

    if single_state: return evals[0]
    else: return evals

