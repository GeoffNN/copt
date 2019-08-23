# python3
"""
Comparison of different step-sizes in Frank-Wolfe
=================================================

Speed of convergence of different step-size strategies
and on 4 different classification datasets.
"""
import copt as cp
import matplotlib.pylab as plt
import numpy as np

# .. datasets and their loading functions ..
datasets = [
    ("Gisette", cp.datasets.load_gisette),
    ("RCV1", cp.datasets.load_rcv1),
    ("Madelon", cp.datasets.load_madelon),
    ("Covtype", cp.datasets.load_covtype)
    ]

variants_fw = [
    ["adaptive", "adaptive step-size"],
    ["adaptive2", "adaptive2 step-size"],
    ["adaptive3", "adaptive3 step-size"],
    [None, "Lipschitz step-size"]]

for dataset_title, load_data in datasets:
  print("Running on the %s dataset" % dataset_title)

  X, y = load_data()
  n_samples, n_features = X.shape

  # the size of the constraint set. We set it to
  # (for example) n_features / 2
  l1_ball = cp.utils.L1Ball(n_features / 2.)
  f = cp.utils.LogLoss(X, y)
  x0 = np.zeros(n_features)

  fw_trace = {label: [] for _, label in variants_fw}
  for step_size, label in variants_fw:

    def trace(kw):
      # store the Frank-Wolfe gap g_t
      fw_trace[label].append(kw["g_t"])

    cp.minimize_frank_wolfe(
        f.f_grad,
        x0,
        l1_ball.lmo,
        callback=trace,
        step_size=step_size,
        lipschitz=f.lipschitz,
    )

  plt.figure()
  for _, label in variants_fw:
    plt.plot(fw_trace[label], label=label)
  plt.yscale("log")
  plt.legend()
  plt.xlabel("number of iterations")
  plt.ylabel("FW gap")
  plt.title(dataset_title)
  plt.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.grid()
  plt.show()