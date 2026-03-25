from sympy import *
from sympy.tensor.array.expressions import ArrayTensorProduct, ArraySymbol


def index_map(indices):
    return {k: j for j, k in enumerate(indices)}


def elasticity(voigt_indices, C_indices):
    s = r"\begin{equation}\begin{aligned}" + "\n"
    for idx, (a, b) in enumerate(C_indices):
        i, j = voigt_indices[a]
        k, ell = voigt_indices[b]
        s += f"C_{{{idx}}} &= C_{{{i}{j},{k}{ell}}}, "
        if idx % 3 == 2:
            s += r"\\" + "\n"
        else:
            s += " & "
    s += r"\end{aligned}\end{equation}" + "\n"
    return s


def stress_strain(voigt_indices, C_indices):
    """Matrix - vector product sigma = C epsilon"""
    voigt_map = index_map(voigt_indices)
    C_map = index_map(C_indices)

    # 21 independent components of the 3x3x3x3 elasticity tensor
    C = ArraySymbol("C", (21,))

    # Independent components of strain tensor in Voigt notation
    epsilon = ArraySymbol("varepsilon", (6,))
    # Independent components of stress tensor in Voigt notation
    sigma = zeros(6)
    s_latex = r"\begin{equation}\begin{aligned}" + "\n"
    s_code = "sigma = jnp.stack(["
    for a in range(6):
        for k in range(3):
            for ell in range(3):
                b = voigt_map[tuple(sorted([k, ell]))]
                _a, _b = sorted((a, b))
                sigma[a] += C[C_map[(_a, _b)]] * epsilon[b]
        s_latex += f"\\sigma_{{{a}}} &= " + latex(simplify(sigma[a])) + r"\\" + "\n"
        s_code += str(simplify(sigma[a])).replace("varepsilon", "epsilon") + ",\n"
    s_latex += r"\end{aligned}\end{equation}" + "\n"
    s_code += "])"
    return s_latex, s_code
    # TODO: generate Jax code for this


def acoustic_matrix(voigt_indices, C_indices, isotropic=False):
    """Components of acoustic matrix"""
    voigt_map = index_map(voigt_indices)
    C_map = index_map(C_indices)
    # Acoustic 3x3 matrix
    K = zeros(3, 3)
    # 21 independent components of the 3x3x3x3 elasticity tensor
    C0 = ArraySymbol("C0", (21,))
    # Lame coefficients for isotropic material
    mu0, lmbda0 = symbols("mu^0 lambda^0")
    # Fourier modes
    xi = ArraySymbol("xi", (3,))
    # xi = Array(symbols(" ".join([f"xi{j}" for j in range(3)])))
    s_latex = r"\begin{equation}\begin{aligned}" + "\n"
    s_code = ""
    for k in range(3):
        for i in range(3):
            for j in range(3):
                for ell in range(3):
                    a, b = sorted(
                        (
                            voigt_map[tuple(sorted((k, j)))],
                            voigt_map[tuple(sorted((i, ell)))],
                        )
                    )
                    K[k, i] += C0[C_map[(a, b)]] * xi[j] * xi[ell]
    s_code += "K = jnp.stack([" + "\n"
    for i in range(3):
        s_code += "jnp.stack([" + "\n"
        for j in range(3):
            s_code += str(K[i, j]) + ",\n"
        s_code += "])," + "\n"
    s_code += "])" + "\n"
    for i in range(3):
        for j in range(i + 1):
            v = K[i, j]
            if isotropic:
                for k in range(3):
                    v = v.subs(C0[k], 2 * mu0 + lmbda0)
                for k in range(3, 6):
                    v = v.subs(C0[k], mu0)
                for k in range(6, 9):
                    v = v.subs(C0[k], lmbda0)
                for k in range(9, 21):
                    v = v.subs(C0[k], 0)
            s_latex += (
                f"K^{0}_{{{i}{j}}} &= "
                + latex(simplify(v))
                .replace("{C_{0}}", "C^{0}")
                .replace(r"\xi", r"\mathring{{\widetilde{{\xi}}}}")
                + "\\\\"
                + "\n"
            )
    s_latex += r"\end{aligned}\end{equation}" + "\n"
    return s_latex, s_code
    # TODO: generate Jax code for this


def fourier_solve(voigt_indices):
    # Fourier modes
    xi = Array(symbols("xi_0 xi_1 xi_2"))
    N00, N01, N02, N11, N12, N22 = symbols(
        "N^{0}_{00} N^{0}_{01} N^{0}_{02} N^{0}_{11} N^{0}_{12} N^{0}_{22}"
    )

    N = Array([[N00, N01, N02], [N01, N11, N12], [N02, N12, N22]])
    N_xi_xi = tensorproduct(N, xi, xi)
    Gamma = Rational(1, 4) * (
        permutedims(N_xi_xi, (3, 0, 1, 2))
        + permutedims(N_xi_xi, (0, 3, 1, 2))
        + permutedims(N_xi_xi, (3, 0, 2, 1))
        + permutedims(N_xi_xi, (0, 3, 2, 1))
    )
    s_latex = r"\begin{equation}\begin{aligned}" + "\n"
    count = 0
    for a in range(6):
        for b in range(a, 6):
            k, ell = voigt_indices[a]
            i, j = voigt_indices[b]
            v = simplify(Gamma[k, ell, i, j])
            s_latex += f"\\widehat{{\\Gamma}}^{{0}}_{{{a}{b}}} &= " + latex(v).replace(
                r"\xi", r"\mathring{{\widetilde{{\xi}}}}"
            )
            if count % 2 == 0:
                s_latex += r", & " + "\n"
            else:
                s_latex += r",\\[1ex]" + "\n"
            count += 1
    s_code = ""
    s_code += "Gamma = jnp.stack([" + "\n"
    for a in range(6):
        s_code += "jnp.stack([" + "\n"
        for b in range(6):
            k, ell = voigt_indices[a]
            i, j = voigt_indices[b]
            v = simplify(Gamma[k, ell, i, j])
            v = str(v)
            for j in range(3):
                v = v.replace(f"xi_{j}", f"xi[{j}]")
            for j in range(3):
                for k in range(3):
                    v = v.replace(f"N^{{0}}_{{{j}{k}}}", f"N[{j},{k}]")
            s_code += v + ",\n"
        s_code += "])," + "\n"
    s_code += "])" + "\n"
    s_latex += r"\end{aligned}\end{equation}" + "\n"
    return s_latex, s_code
    # TODO: generate Jax code for this


# Voigt-indices
#
#  0 3 4
#    1 5
#      2
voigt_indices = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

# Indices used for numbering the 21 indendendent components of the elasticity tensor
# when written down in Voigt notation

#  0  6  7  9 10 11
#     1  8 12 13 14
#        2 15 16 17
#           3 18 19
#              4 20
#                 5

C_indices = (
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5),
    (4, 5),
)
with open("anisotropic_elasticity.tex", "w", encoding="utf8") as f:
    print(r"% ---- elasticity tensor ----", file=f)
    s_latex = elasticity(voigt_indices, C_indices)
    print(s_latex, file=f)

s_latex, s_code = stress_strain(voigt_indices, C_indices)
with open("anisotropic_stress_strain.tex", "w", encoding="utf8") as f:
    print(r"% ---- sigma_{ij} = C_{ijkl} epsilon_{kl} ----", file=f)
    print(s_latex, file=f)
with open("anisotropic_stress_strain.py", "w", encoding="utf8") as f:
    print(s_code, file=f)

s_latex, s_code = acoustic_matrix(voigt_indices, C_indices, isotropic=False)
with open("anisotropic_acoustic_matrix.tex", "w", encoding="utf8") as f:
    print(r"% ---- acoustic matrix K^0 ----", file=f)
    print(s_latex, file=f)

with open("anisotropic_acoustic_matrix.py", "w", encoding="utf8") as f:
    print(s_code, file=f)

s_latex, s_code = acoustic_matrix(voigt_indices, C_indices, isotropic=True)
with open("isotropic_acoustic_matrix.tex", "w", encoding="utf8") as f:
    print(r"% ---- acoustic matrix K^0 [isotropic material] ----", file=f)
    print(s_latex, file=f)

s_latex, s_code = fourier_solve(voigt_indices)
with open("anisotropic_fourier_matrix.tex", "w", encoding="utf8") as f:
    print(r"% ---- Fourier solve ----", file=f)
    print(s_latex, file=f)
with open("anisotropic_fourier_matrix.py", "w", encoding="utf8") as f:
    print(s_code, file=f)
