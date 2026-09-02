> ### ⚠ Read this before comparing these numbers to any other page
>
> **This run reports 3 seeds, not 5, and the two missing seeds are missing because they died.**
>
> Seeds 2 and 5 both aborted mid-training with an exploding control. The failure is
> identical in each case, raised from `project_theta_to_sigma` inside
> `controls/specific_entropy_matrix.py`:
>
> ```
> RuntimeError: eigh failed AND Theta is not finite -- this is an exploding control,
> not a cuSOLVER convergence problem. Do not retry on another backend.
> Theta: shape=(256, 8, 8), dtype=torch.float32, non_finite_entries=16384,
> bad_batch_elements=[0, 1, 2, 3, 4, 5, 6, 7]... of 256
> ```
>
> `non_finite_entries=16384` is exactly `256 x 8 x 8` -- the *entire* batch of Cholesky
> factors went non-finite, not a marginal eigenvalue that spectral clipping could have
> caught. The last rows written to `losses/seed_2_losses.csv` and
> `losses/seed_5_losses.csv` are step 1600 and step 1500; neither reached the 2500 steps
> the surviving seeds completed. Their loss curves contain no NaN, because the run dies
> inside the eigendecomposition before a bad loss is ever logged -- the truncated CSV is
> the only trace left in the loss history, which is why the convergence figure below
> shows three curves and not five.
>
> **Two consequences, both of which make this page non-comparable to its siblings:**
>
> 1. **A 2-in-5 divergence rate at this budget is itself the headline result**, and it is
>    not visible anywhere in the metric tables below. Those tables describe only the seeds
>    that survived, which is a conditional-on-survival sample, not a random one.
> 2. **N = 3 inflates every standard deviation** relative to the 5-seed campaigns. The
>    dataset-level leaderboard in [`../oldreadme.md`](../oldreadme.md) counts how many
>    rows sit at or below the independent-draw floor, and a wider std widens the tie
>    band, so this run's row count is mechanically favoured. **Do not read its position
>    on that leaderboard as evidence that the quarter budget is competitive.**
>
> The surviving seeds are 0, 4 and 6. Every "Seed *n*" column below is labelled with its
> true seed id, not its position in the table.
