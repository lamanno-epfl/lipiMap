A few tips on training models
===================

lipiMap
  .. - The main hyperparameter that affects the quality of integration for the reference training is alpha_kl, the value of which is multiplied by the kl divergence term in the total loss.
  .. - If the visualized latent space looks like a single blob after the reference training, we recommend to decrease the value of alpha_kl. If the visualized latent space shows bad integration quality, we recommend to increase the value of alpha_kl. The good default value in most cases is alpha_kl = 0.5.
  .. - The required strength of group lasso regularization (alpha) depends on the number of used LPs and the size of the dataset.
  .. - If soft mask in the reference training is used (`soft_ext_mask=True` in the model initialization), it is better to start with `alpha_l1=0.5` (higher value means more constraints on how many features are added to the sets) and use `print_stats=True` in the training for monitoring to check the reported "Share of deactivated inactive features: ​__"  is around 95% (0.95) at the end and stays so at the final 10 epochs of training. If it is much smaller, `alpha_l1` should be increased by a small value (around 0.05), and if it is 100% (1.) then alpha_l1 should be decreased.
  .. - Using new terms (`n_ext`) in the reference training is not recommended.
