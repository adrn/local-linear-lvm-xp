"""
Pick a few quality check neighborhoods
Run optimization with different [learning rate, n_latents]
"""

import sys
import numpy as np
import h5py

from jax.config import config as jax_config
jax_config.update("jax_enable_x64", True)
import jaxopt
import optax

import schlummernd as sch
from schlummernd.lvm import LinearLVM


def worker(task):
    idx, task = task

    with h5py.File(task['filename'], 'r') as f:
        if str(idx) in f.keys():
            return

    f = task['f']
    lbl = task['lbl']

    rng = np.random.default_rng(seed=task['seed'])
    rando = rng.integers(4, size=len(f))

    n_labels = len(lbl.labels)
    n_latents = n_labels + task['n_latents']
    if 'schmag' in lbl.labels:
        schmag_i = list(lbl.labels.keys()).index('schmag')
    else:
        schmag_i = None

    all_idx = []
    all_predict = []
    all_true_y = []
    all_true_yerr = []
    for i in np.unique(rando):
        train = rando != i
        valid = ~train

        f_train = f[train]
        f_valid = f[valid]

        X_train, X_valid = f_train.X, f_valid.X
        X_train_err, X_valid_err = f_train.X_err, f_valid.X_err
        y_train, y_valid = lbl.y[train], lbl.y[valid]
        y_train_err, y_valid_err = lbl.y_err[train], lbl.y_err[valid]

        kwargs = dict()
        if 'schmag' in lbl.labels:
            kwargs['B_fit_idx'] = [schmag_i]

        llvm = LinearLVM(
            X_train, y_train,
            X_train_err, y_train_err,
            n_latents,
            alpha=task['ab'],
            beta=task['ab'],
            verbose=False,
            rng=rng,
            **kwargs
        )
        x0 = llvm.pack_p()

        opt = optax.adam(task['learning_rate'])
        solver = jaxopt.OptaxSolver(opt=opt, fun=llvm, maxiter=2**18)
        res_adam = solver.run(x0)

        initial_loss = llvm(x0)
        final_loss = llvm(res_adam.params)
        assert final_loss < initial_loss

        # Test on validation sample:
        res_state = llvm.unpack_p(res_adam.params)

        y_valid_predict = llvm.predict_y(
            X_valid,
            X_valid_err,
            res_state
        )
        all_predict.append(y_valid_predict)
        all_idx.append(np.where(valid)[0])
        all_true_y.append(y_valid)
        all_true_yerr.append(y_valid_err)

    result = {}
    result['predict_y'] = lbl._untransform(np.concatenate(all_predict))
    result['true_y'] = lbl._untransform(np.concatenate(all_true_y))
    result['true_yerr'] = lbl._untransform(np.concatenate(all_true_yerr), err=True)
    result['filename'] = task['filename']
    result['idx'] = idx
    all_idx = np.concatenate(all_idx)

    return result


def callback(result):

    with h5py.File(result['filename'], 'r+') as f:
        grp = f.create_group(str(result['idx']))
        for k, data in result.items():
            if k == 'filename':
                continue
            grp.create_dataset(name=k, data=data)


def main(pool):
    conf = sch.Config.parse_yaml('../config.yml')

    output_path = conf.output_path / 'hyperpar'
    output_path.mkdir(exist_ok=True)

    g_all = conf.load_training_data()

    # Load the neighborhoods: An array of index arrays
    hoods = np.load(
        conf.data_path / 'training_neighborhoods.npy',
        allow_pickle=True
    )

    # These are neighborhood indices that span a range of properties, to use as guides
    # for setting the hyperparameters
    tasks = []
    for hood_n in [2]:
        g = g_all[hoods[hood_n]]
        g = g.filter(
            H_ERR=(0, 0.2)
        )

        # add G-J color as an additional feature
        other_features = {
            r"$G-J$": (
                0.25 * (g.phot_g_mean_mag.value - g.J),
                0.25 * np.sqrt(1/g.phot_g_mean_flux_over_error**2 + g.J_ERR**2)
            )
        }
        f_all = sch.Features.from_gaiadata(g, n_bp=25, n_rp=25, **other_features)

        # Set up labels:
        lbl = sch.Labels()

        schmag_factor = 10 ** (0.2 * g.phot_g_mean_mag.value) / 100.
        lbl.add_label(
            'schmag',
            value=g.parallax.value * schmag_factor,
            err=g.parallax_error.value * schmag_factor,
            label='$G$-band schmag [absmgy$^{-1/2}$]'
        )
        lbl.add_label(
            'TEFF',
            g.TEFF,
            g.TEFF_ERR,
            label=r"$T_{\rm eff}$ [K]"
        )
        lbl.add_label(
            'M_H',
            g.M_H,
            g.M_H_ERR,
            label=r"$[{\rm M}/{\rm H}]$"
        )
        lbl.add_label(
            'logg',
            g.LOGG,
            g.LOGG_ERR,
            label=r"$\log g$"
        )
        lbl.add_label(
            'EBV',
            g.SFD_EBV,
            np.sqrt(0.01**2 + (0.02 * np.abs(g.SFD_EBV))**2),
            label=r"E$(B-V)$"
        )
        assert np.all(lbl.y_err > 0)

        for ab in [1e-2, 1e-1, 1e0]:
            for n_latents in [1, 4, 8]:
                for learning_rate in [1e-6, 1e-4, 1e-2]:
                    task = {

                        'ab': ab,
                        'n_latents': n_latents,
                        'learning_rate': learning_rate,
                        'lbl': lbl,
                        'f': f_all,
                        'filename': output_path / f'hyperpars-{hood_n:04d}.hdf5'
                    }
                    tasks.append(task)

    for res in pool.map(worker, tasks, callback=callback):
        pass


if __name__ == '__main__':
    from schwimmbad import MPIPool

    with MPIPool() as pool:
        main(pool)

    sys.exit(0)
