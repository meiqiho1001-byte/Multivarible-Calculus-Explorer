import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr  
import numpy as np
import plotly.graph_objects as go
import re

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Calculus Intelligence Pro", layout="wide")

# ---------------------------
# STYLE 
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

/* Background and Font */
.stApp {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Inter', times new roman;
}

.topic-card {
    background: #161b22;
    padding: 25px;
    border-radius: 12px;
    margin-bottom: 25px;
    border: 1px solid #30363d;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}

.topic-header {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 15px;
    padding: 5px 0 5px 15px;
    display: flex;
    align-items: center;
}

/* (Accent Bars) */
.geom-h { border-left: 5px solid #38bdf8; color: #38bdf8; }     /*blue*/
.grad-h { border-left: 5px solid #22c55e; color: #22c55e; }    /*green*/
.partial-h { border-left: 5px solid #facc15; color: #facc15; } /*yellow*/
.theory-h { border-left: 5px solid #a78bfa; color: #a78bfa; }  /*purple*/
.critical-h { border-left: 5px solid #f472b6; color: #f472b6; }/*pink*/

.explanation-text {
    color: #8b949e;
    font-size: 0.95rem;
    line-height: 1.6;
    margin-bottom: 15px;
}

.stTextInput input, .stSelectbox div {
    background-color: #010409 !important;
    border: 1px solid #30363d !important;
    color: white !important;
}

hr {
    border: 0;
    height: 1px;
    background: #30363d;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# PARSER & LOGIC
# ---------------------------
def robust_math_parse(text):
    text = text.lower().replace(" ", "")
    text = re.sub(r'(\d)([a-z\(])', r'\1*\2', text)
    text = re.sub(r'([x-z])([a-z\(])', r'\1*\2', text)
    return text.replace("^", "**")

MATH_MAP = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "sqrt": sp.sqrt, "log": sp.log}

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.header("📍 Domain Space")
    col_x1, col_x2 = st.columns(2)
    x_min = col_x1.number_input("X Min", value=-5.0)
    x_max = col_x2.number_input("X Max", value=5.0)
    col_y1, col_y2 = st.columns(2)
    y_min = col_y1.number_input("Y Min", value=-5.0)
    y_max = col_y2.number_input("Y Max", value=5.0)
    col_z1, col_z2 = st.columns(2)
    z_min = col_z1.number_input("Z Min", value=-5.0)
    z_max = col_z2.number_input("Z Max", value=5.0)

# ---------------------------
# MAIN INTERFACE
# ---------------------------
st.title("Multivariable Calculus Explorer")
user_raw = st.text_input("Define your function f(x, y, z):", value="sin(2*x) + 5*cos(y) - 8*z")

x, y, z = sp.symbols("x y z")
try:
    processed = robust_math_parse(user_raw)
    f_sym = parse_expr(processed, local_dict=MATH_MAP)

    vars_present = f_sym.free_symbols
    if 'z' in processed:
        vars_present = vars_present | {z}
    if 'x' in processed:
        vars_present = vars_present | {x}
    if 'y' in processed:
        vars_present = vars_present | {y}

except Exception as e:
    st.error(f"Syntax Error: {e}")
    st.stop()

fx = sp.diff(f_sym, x)
fy = sp.diff(f_sym, y)
fz = sp.diff(f_sym, z)

col_left, col_right = st.columns([3,2])

# ---------------------------
# I. GEOMETRY
# ---------------------------
with col_left:
    st.markdown('<div class="topic-card">', unsafe_allow_html=True)
    st.markdown('<div class="topic-header geom-h">I. Geometric Meaning & Visualization</div>', unsafe_allow_html=True)

    if z in vars_present:
        st.markdown('<div class="explanation-text"><b>Function of Three Variables:</b> This represents a scalar field. We use a level surface (z-slice) to visualize the topography at a specific altitude.</div>', unsafe_allow_html=True)
        z_slice = st.select_slider("Adjust Z-Slice Position", options=np.round(np.linspace(z_min, z_max, 21), 2), value=0.0)
        f_plot = f_sym.subs(z, z_slice)
    else:
        st.markdown('<div class="explanation-text"><b>Function of Two Variables:</b> A standard 3D surface mapping (x,y) to height.</div>', unsafe_allow_html=True)
        f_plot = f_sym

    xv = np.linspace(x_min, x_max, 50)
    yv = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(xv, yv)
    fn = sp.lambdify((x, y), f_plot, "numpy")
    Z_vals = fn(X,Y)
    if np.isscalar(Z_vals): Z_vals = np.full(X.shape, Z_vals)
    Z_vals = np.real(Z_vals)
    Z_vals[~np.isfinite(Z_vals)] = np.nan

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z_vals, colorscale="IceFire")])
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,b=0,t=0), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    z_low, z_high = np.nanmin(Z_vals), np.nanmax(Z_vals)
    st.write(f"**Domain:** $x \in [{x_min}, {x_max}], y \in [{y_min}, {y_max}]$")
    st.write(f"**Observed Range:** $f \in [{z_low:.2f}, {z_high:.2f}]$")

    if any(trig in user_raw.lower() for trig in ["sin","cos","tan"]):
        st.markdown("---")
        st.write("**Shape:** Periodic/Wave Surface. The function oscillates between high and low points.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# II. GRADIENT
# ---------------------------
with col_right:
    st.markdown('<div class="topic-card">', unsafe_allow_html=True)
    st.markdown('<div class="topic-header grad-h">II. Gradient & Steepest Ascent</div>', unsafe_allow_html=True)
    st.markdown('<div class="explanation-text">The <b>Gradient</b> ($\\nabla f$) is a vector field that lives in the domain of the function. Its most critical property is that at any given point, the gradient vector points in the <b>Direction of Steepest Ascent</b>. If you were standing on this surface, following the gradient would be the most efficient path to go uphill. The magnitude of the gradient tells you exactly how steep that slope is.</div>', unsafe_allow_html=True)

    
    g_comps = [sp.latex(fx), sp.latex(fy)]
    if z in vars_present: g_comps.append(sp.latex(fz))
    st.latex(rf"\nabla f = \langle {', '.join(g_comps)} \rangle")
    
    st.write("**Geometric Meaning:**")
    st.markdown("- **Direction:** Points toward local maxima.")
    st.markdown("- **Orthogonality:** The gradient is always perpendicular to the level curves (contours).")
    st.markdown("- **Rate of Change:** $|\nabla f|$ is the maximum possible directional derivative.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# III. PARTIAL DERIVATIVES
# ---------------------------
st.markdown('<div class="topic-card">', unsafe_allow_html=True)
st.markdown('<div class="topic-header partial-h">III. Partial Derivative Explorer</div>', unsafe_allow_html=True)

p_col1, p_col2 = st.columns(2)
with p_col1:
    var_choice = st.selectbox("**Differentiable with respect to:**", [v for v in ['x','y','z'] if sp.Symbol(v) in vars_present], key="var_sel")
with p_col2:
    order = st.radio("**Order of derivative:**", [1,2], horizontal=True, key="order_sel")

var = sp.Symbol(var_choice)
partial = sp.diff(f_sym, var, order)
st.write("**Resulting Partial Derivative:**")
st.latex(rf"\frac{{\partial^{order} f}}{{\partial {var_choice}^{order}}} = {sp.latex(partial)}")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# IV. TOTAL DIFFERENTIAL
# ---------------------------
t_col1, t_col2 = st.columns(2)

with t_col1:
    st.markdown('<div class="topic-card">', unsafe_allow_html=True)
    st.markdown('<div class="topic-header theory-h">IV. Total Differential (df)</div>', unsafe_allow_html=True)
    st.markdown('<div class="explanation-text">The total differential represents the principal part of the change in a function with respect to changes in the independent variables.</div>', unsafe_allow_html=True)
    df_expr = rf"df = \left( {sp.latex(fx)} \right)dx + \left( {sp.latex(fy)} \right)dy"
    if z in vars_present: df_expr += rf" + \left( {sp.latex(fz)} \right)dz"
    st.latex(df_expr)
    st.markdown('</div>', unsafe_allow_html=True)

with t_col2:
    st.markdown('<div class="topic-card">', unsafe_allow_html=True)
    st.markdown('<div class="topic-header theory-h">V. Differentiability</div>', unsafe_allow_html=True)
    st.markdown('<div class="explanation-text">Continuity and differentiability are fundamental properties. A function differentiable at a point is always continuous there.</div>', unsafe_allow_html=True)

elementary_funcs = {sp.sin, sp.cos, sp.tan, sp.exp, sp.log, sp.sqrt}

is_elementary = all(
    isinstance(node, (sp.Add, sp.Mul, sp.Pow, sp.Symbol, sp.Number)) 
    or node.func in elementary_funcs
    for node in sp.preorder_traversal(f_sym)
)

if is_elementary:
    st.write("✅ **Continuity:** Function is elementary → continuous everywhere.")
    st.write("✅ **Differentiability:** All partial derivatives exist → differentiable everywhere.")
else:
    st.write("⚠️ *General Case:* Differentiability not guaranteed. Further analysis needed.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# VI. CRITICAL POINTS
# ---------------------------
st.markdown('<div class="topic-card">', unsafe_allow_html=True)
st.markdown('<div class="topic-header critical-h">VI. Critical Points & Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="explanation-text">A critical point occurs when a function derivative is zero or undefined, indicating a potential local maximum or minimum, or a saddle point where the slope changes direction. </div>', unsafe_allow_html=True)

eqs = []
vars_ = []

if x in vars_present:
    eqs.append(fx); vars_.append(x)
if y in vars_present:
    eqs.append(fy); vars_.append(y)
if z in vars_present:
    eqs.append(fz); vars_.append(z)
    
try:
    sols = sp.solve(eqs, vars_, dict=True)

    if sols:
        for s in sols[:3]:
            fxx = sp.diff(f_sym, x, 2).subs(s) if x in vars_present else None
            fyy = sp.diff(f_sym, y, 2).subs(s) if y in vars_present else None
            fxy = sp.diff(f_sym, x, y).subs(s) if {x,y}.issubset(vars_present) else None

            if fxx is not None and fyy is not None and fxy is not None:
                D = fxx*fyy - fxy**2
                label = "Saddle Point" if D < 0 else ("Local Minimum" if fxx > 0 else "Local Maximum")
            else:
                label = "Critical Point"

            st.write(f"📍 **Point {s}** — **{label}**")
    else:
        st.info("No stationary points found.")
except:
    st.warning("Analytical solution too complex for real-time solver.")

st.markdown('</div>', unsafe_allow_html=True)
