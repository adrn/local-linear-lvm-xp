import astropy.table as at
import numpy as np

from schlummernd.data import Features, Labels


class TestFeatures:
    def setup(self):
        rng = np.random.default_rng(seed=42)
        self.rng = rng
        N_fake = 64
        self.data = at.Table()

        N_bp = 5
        self.data["bp"] = rng.uniform(5, 10.0, size=(N_fake, N_bp))

        _cov = rng.uniform(0.1, 0.5, size=(N_fake, N_bp, N_bp)) ** 2
        _cov = np.einsum("nij,nik->njk", _cov, _cov)
        self.data["bp_cov"] = _cov
        self.data["bp_err"] = np.sqrt(_cov[:, np.arange(N_bp), np.arange(N_bp)])
        self.data["bp_icov"] = np.linalg.inv(_cov)

        self.data["G-J"] = rng.uniform(12, 18.0, size=N_fake)
        self.data["G-J_ERR"] = rng.uniform(0, 0.5, size=N_fake)

    def make_obj(self, err_input):
        f = Features()

        f.add("G-J", value=self.data["G-J"], err=self.data["G-J_ERR"])

        if err_input == "err":
            f.add("bp", value=self.data["bp"], err=self.data["bp_err"])
        elif err_input == "cov":
            f.add("bp", value=self.data["bp"], cov=self.data["bp_cov"])
        elif err_input == "icov":
            f.add("bp", value=self.data["bp"], icov=self.data["bp_icov"])

        return f

    def test_api_shit(self):
        f1 = self.make_obj("err")
        f2 = self.make_obj("cov")
        f3 = self.make_obj("icov")

        new_f1 = f1[3:7]
        assert len(new_f1) == 4

        data = f1._make_helper(icov=False)
        data, icov = f1._make_helper()

    def test_different_errs(self):
        N = 15
        M = 3
        val = self.rng.uniform(-1, 1, size=(N, M))
        err = self.rng.uniform(0.1, 0.2, size=(N, M))

        cov = np.zeros((N, M, M))
        cov[:, np.arange(M), np.arange(M)] = err**2

        icov = np.zeros((N, M, M))
        icov[:, np.arange(M), np.arange(M)] = 1 / err**2

        f1 = Features()
        f1.add("bp", value=val, err=err)

        f2 = Features()
        f2.add("bp", value=val, cov=cov)

        f3 = Features()
        f3.add("bp", value=val, icov=icov)

        assert np.allclose(f1._raw_data["bp"], f1._raw_data["bp"])
        assert np.allclose(f1._raw_data["bp"], f3._raw_data["bp"])

        assert np.allclose(f1._raw_icov["bp"], f1._raw_icov["bp"])
        assert np.allclose(f1._raw_icov["bp"], f3._raw_icov["bp"])


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

    def make_lbl(self):
        lbl = Labels()

        lbl.add(
            "parallax",
            self.data["parallax"],
            err=self.data["parallax_error"],
            plot_label=r"$\varpi",
        )
        lbl.add(
            "PARAM",
            self.data["PARAM"],
            cov=self.data["PARAM_COV"],
        )
        lbl.add(
            "J",
            self.data["J"],
            icov=1 / self.data["J_ERR"] ** 2,
            plot_label=r"$J$",
        )
        return lbl

    def test_api(self):
        lbl = self.make_lbl()

        # unit tests of lbl attribute shapes:
        y, y_Cinv = lbl.make_y(icov=True)
        assert y.shape == (len(self.data), 4)
        assert y_Cinv.shape == (len(self.data), 4, 4)
        assert len(lbl.plot_labels) == 3

    # TODO: update
    # def test_roundtrip(self):
    #     lbl = self.make_lbl()
    #     lbl2 = lbl.from_transformed(lbl.y, lbl.y_Cinv)

    #     for key in lbl._vals:
    #         assert np.allclose(lbl._vals[key], lbl2._vals[key])

    #     assert np.allclose(lbl.y, lbl2.y)
    #     assert np.allclose(lbl.y_Cinv, lbl2.y_Cinv)

    #     # Make sure things are un-cached properly:
    #     lbl3 = lbl.from_transformed(lbl.y + 1.1, lbl.y_Cinv + 1e-3)

    #     for key in lbl._vals:
    #         assert not np.allclose(lbl._vals[key], lbl3._vals[key])

    #     assert not np.allclose(lbl.y, lbl3.y)
    #     assert not np.allclose(lbl.y_Cinv, lbl3.y_Cinv)

    def test_slice(self):
        lbl = self.make_lbl()

        lbl2 = lbl[3:10]
        lbl3 = lbl[np.arange(3, 10)]

        assert lbl2.N == lbl3.N == 7
        y2 = lbl2.make_y(icov=False)
        y3 = lbl3.make_y(icov=False)
        assert y2.shape[0] == y3.shape[0] == 7
