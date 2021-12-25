from ..experiment import NHsqc, Hsqc


class Gramicidin():
    """
    40 mM in DMSO. 15N HSQC, 13C HSQC data available.
    """
    nhsqc_peaks = [(127.9990, 9.0926),  # Phe NH
                   (125.4206, 8.6639),  # Orn NH
                   (123.2816, 8.3292),  # Leu NH
                   (113.2316, 7.2254)   # Val NH
                   ]  # Orn epsilon-NH2 is folded in most spectra.
    nhsqc = NHsqc(nhsqc_peaks)

    hsqc_ch = [(129.8390, 7.2606),  # Phe ortho
               (128.7822, 7.2900),  # Phe meta
               (127.3291, 7.2488),  # Phe para
               (60.3538, 4.3071),   # Pro alpha
               (57.3154, 4.4070),   # Val alpha
               (54.4092, 4.3600),   # Phe alpha
               (51.3709, 4.7651),   # Orn alpha
               (50.0499, 4.5714),   # Leu alpha
               (31.5556, 2.0759),   # Val beta
               (24.4222, 1.4065),   # Leu delta
               ]
    hsqc_ch2 = [(46.4831, 3.5908), (46.4831, 2.4987),  # Pro delta
                (41.4633, 1.3537), (41.4633, 1.3008),  # Leu beta
                (39.0855, 2.8392), (39.0855, 2.7805),  # Orn delta
                (36.1792, 2.9743), (36.1792, 2.8803),  # Phe beta
                (30.1026, 1.7471), (30.1026, 1.6062),  # Orn beta
                (29.4421, 1.9526), (29.4421, 1.4761),  # Pro beta
                (23.6296, 1.5122),  # Pro gamma
                (23.4975, 1.6473),  # Orn gamma
                ]
    hsqc_ch3 = [(23.1012, 0.8018),  # Leu delta
                (19.4024, 0.7665),  # Val gamma
                (18.4777, 0.8076),  # Val gamma
                ]
    hsqc_margin = (0.5, 0.02)
    hsqc = Hsqc(hsqc_ch, hsqc_ch2, hsqc_ch3, hsqc_margin)
