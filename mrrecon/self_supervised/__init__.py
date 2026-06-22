"""Self-supervised reconstruction trained across the dataset WITHOUT ground truth.

Each slice's acquisition mask Omega is split into a data-consistency set (Theta)
and a loss set (Lambda); the network is trained on the k-space error at Lambda.

Algorithms, selected with ``--algo``:
    ssdu  -- SSDU (Yaman et al.); regulariser chosen with ``--model``
    sscu  -- SSCU (stub, not implemented yet)

Run:  python -m mrrecon.self_supervised --algo ssdu --tissue knee --run_name ssdu1
"""
