import astropy.table as at
import numpy as np

from schlummernd.data import Labels


class TestLabels:
    def setup(self):
        rng = np.random.default_rng(seed=42)
        N_fake = 64
        self.data = at.Table()
        self.data["parallax"] = rng.uniform(0, 10.0, size=N_fake)
        self.data["parallax_error"] = rng.uniform(0, 1.0, size=N_fake)

        self.data["PARAM"] = np.zeros((N_fake, 2))
        self.data["PARAM"][:, 0] = rng.normal(5000, 150, size=N_fake)
        self.data["PARAM"][:, 1] = rng.uniform(2.5, 3, size=N_fake)

        Ctmp = np.zeros((N_fake, 2, 2))
        err1 = rng.uniform(10, 250, size=N_fake)
        err2 = rng.uniform(0.05, 0.2, size=N_fake)
        r12 = rng.uniform(-1, 1, size=N_fake)
        Ctmp[:, 0, 0] = err1**2
        Ctmp[:, 1, 1] = err2**2
        Ctmp[:, 0, 1] = Ctmp[:, 1, 0] = r12 * err1 * err2
        self.data["PARAM_COV"] = Ctmp

        self.data["J"] = rng.uniform(12, 18.0, size=N_fake)
        self.data["J_ERR"] = rng.uniform(0, 0.5, size=N_fake)

    def test_api(self):
        lbl = Labels()

        lbl.add_label(
            "parallax",
            self.data["parallax"],
            self.data["parallax_error"] ** 2,
            plot_label=r"$\varpi",
        )
        lbl.add_label_group(
            ["TEFF", "logg"],
            self.data["PARAM"],
            cov=self.data["PARAM_COV"],
            plot_labels=[r"$T_{\rm eff}$", r"$\log g$"],
        )

        lbl.add_label(
            "J",
            self.data["J"],
            self.data["J_ERR"] ** 2,
            plot_label=r"$J$",
        )

        # unit tests of lbl attribute shapes:
        assert lbl.y.shape == (len(self.data), 4)
        assert lbl.y_Cinv.shape == (len(self.data), 4, 4)
        assert len(lbl.plot_labels) == 4
