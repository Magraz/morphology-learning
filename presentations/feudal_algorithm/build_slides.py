#!/usr/bin/env python3
"""Build a matched PowerPoint / PDF deck, with editable text and diagrams.

Requires python-pptx, reportlab, matplotlib and Pillow. Equations are rendered
at 240 dpi; their editable LaTeX is retained in the source and speaker notes.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/feudal-slides-matplotlib")
import matplotlib
matplotlib.use("Agg")
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
from PIL import Image
from pptx import Presentation
from pptx.util import Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "equations"
ASSETS.mkdir(exist_ok=True)
W, H = 960, 540
BG, INK, MUTED = "F6F8FB", "152D45", "526477"
TEAL, PURPLE, AMBER = "087F83", "7055A5", "A96812"
PALE, LAVENDER, GOLD, LINE = "E5F3F1", "EEEAF7", "FBF1DC", "D8E0E8"
WHITE = "FFFFFF"
FONT = "Liberation Sans"
FONT_DIR = Path("/usr/share/fonts/truetype/liberation")
if not FONT_DIR.exists():
    FONT_DIR = Path("/usr/share/fonts/truetype/liberation2")
for suffix, filename in [("", "LiberationSans-Regular.ttf"), ("-Bold", "LiberationSans-Bold.ttf")]:
    pdfmetrics.registerFont(TTFont("Deck" + suffix, str(FONT_DIR / filename)))

SLIDES = []

def wrap(value, width, size, bold=False):
    font = "Deck-Bold" if bold else "Deck"
    lines = []
    for paragraph in value.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = (line + " " + word).strip()
            if pdfmetrics.stringWidth(candidate, font, size) > width and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines

class Slide:
    def __init__(self, section, title, takeaway, notes, source, appendix=False):
        self.section, self.title, self.notes, self.source = section, title, notes, source
        self.elements, self.equations = [], []
        self.number = len(SLIDES) + 1
        self.appendix = appendix
        SLIDES.append(self)
        self.rect(0, 0, W, H, BG)
        self.text(section.upper(), 42, 25, 870, 11, TEAL, True)
        self.text(title, 42, 50, 880, 31, INK, True, max_h=77)
        if takeaway:
            self.rect(42, 467, 876, 37, PALE, radius=6)
            self.text(takeaway, 56, 477, 846, 15, TEAL, True, max_h=24)
        self.text(source, 42, 518, 825, 8.2, MUTED)
        self.text(f"{self.number:02d}", 874, 516, 44, 11, MUTED, align="right")

    def rect(self, x, y, w, h, fill=WHITE, stroke=None, radius=0):
        self.elements.append(dict(kind="rect", x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,radius=radius))

    def text(self, value, x, y, w, size=20, color=INK, bold=False, align="left", max_h=None):
        lines = wrap(value,w,size,bold)
        height = len(lines)*size*1.18 + 5
        if max_h is not None and height > max_h:
            raise ValueError(f"Slide {self.number}: text too tall ({height:.1f} > {max_h}): {value}")
        if y + height > H + 3:
            raise ValueError(f"Slide {self.number}: text off page: {value}")
        self.elements.append(dict(kind="text",text="\n".join(lines),x=x,y=y,w=w,h=height,size=size,color=color,bold=bold,align=align))
        return height

    def eq(self, latex, x, y, w=860, h=58, size=27, color=INK):
        key = hashlib.sha256((latex+str(size)+color).encode()).hexdigest()[:18]
        path = ASSETS / f"{key}.png"
        if not path.exists():
            math_to_image("$"+latex+"$",str(path),prop=FontProperties(size=size),dpi=240,format="png",color="#"+color)
        iw,ih=Image.open(path).size
        natural_w,natural_h=iw*72/240,ih*72/240
        scale=min(w/natural_w,h/natural_h,1.0)
        dw,dh=natural_w*scale,natural_h*scale
        self.elements.append(dict(kind="image",path=path,x=x,y=y+(h-dh)/2,w=dw,h=dh))
        self.equations.append(latex)

    def line(self, x1,y1,x2,y2,color=LINE,width=2,arrow=False):
        self.elements.append(dict(kind="line",x1=x1,y1=y1,x2=x2,y2=y2,color=color,width=width))
        if arrow:
            a=math.atan2(y2-y1,x2-x1)
            points=[(x2,y2),(x2-9*math.cos(a)+4*math.sin(a),y2-9*math.sin(a)-4*math.cos(a)),(x2-9*math.cos(a)-4*math.sin(a),y2-9*math.sin(a)+4*math.cos(a))]
            self.elements.append(dict(kind="polygon",points=points,fill=color))

    def circle(self,x,y,r,fill):
        self.elements.append(dict(kind="circle",x=x-r,y=y-r,w=2*r,h=2*r,fill=fill))

    def card(self, x,y,w,h,label,body,color=TEAL,fill=WHITE,body_size=20):
        self.rect(x,y,w,h,fill,LINE,8)
        self.rect(x,y,5,h,color,radius=0)
        self.text(label,x+18,y+16,w-36,12,color,True)
        self.text(body,x+18,y+44,w-36,body_size,max_h=h-49)

    def bullets(self, items, x=48,y=140,w=860,size=21,gap=16,color=TEAL):
        for item in items:
            self.circle(x+4,y+10,3,color)
            height=self.text(item,x+20,y,w-20,size)
            y+=height+gap

    def table(self, headers, rows, widths, x=42,y=140,row_h=48,size=17):
        total=sum(widths)
        self.rect(x,y,total,35,INK,radius=4)
        px=x
        for txt,ww in zip(headers,widths):
            self.text(txt,px+12,y+10,ww-24,12,WHITE,True)
            px+=ww
        for i,row in enumerate(rows):
            yy=y+35+i*row_h
            self.rect(x,yy,total,row_h,WHITE if i%2==0 else "EDF1F6")
            px=x
            for txt,ww in zip(row,widths):
                self.text(txt,px+12,yy+10,ww-24,size,max_h=row_h-8)
                px+=ww


# 01 — Orientation
s=Slide("Feudal MAPPO • technical walkthrough", "Learning what to ask for\nand how to act", "A manager learns useful goals. A worker learns goal-conditioned actions.",
"This deck explains the current morphology-learning implementation, not a generic textbook version of FeUdal Networks. Plan for roughly 25 minutes for the main slides, with the appendices available for questions. Start with the shared architecture and its losses; then explain each change as an independent design choice. No performance ranking is claimed. All current feudal model groups, including the newly present c80 waypoint presets, are covered by the variant families.",
"morphology-learning / algorithms/feudal_mappo_jax • implementation snapshot: 24 September 2026")
s.text("BASE ALGORITHM → LOSSES & UPDATES → VARIANTS",44,147,870,14,MUTED,True)
s.card(44,213,260,182,"MANAGER", "Which direction\nwould help the task?",PURPLE,LAVENDER,25)
s.card(351,213,260,182,"WORKER", "Which action\nwill make progress?",TEAL,PALE,25)
s.card(658,213,260,182,"ENVIRONMENT", "What changed,\nand was it useful?",AMBER,GOLD,25)
s.line(307,305,344,305,PURPLE,3,True)
s.line(614,305,651,305,TEAL,3,True)

# 02 — Architecture
s=Slide("01 / base algorithm", "Two levels, one environment", "The worker is shared across agents; the manager is centralized even at execution.",
"At each step the manager reads the global state X and constructs a latent s_i and goal g_i for every agent. The default core is an MLP. The worker receives only its local observation plus its goal conditioning, but those goals carry centralized information. The worker uses one shared policy across agents. Centralized critics read X. Continuous policies produce Gaussian parameters; discrete policies produce masked categorical logits. A primitive action means one action in the selected environment; macro environments have their own decision time unit.",
"Sources: manager.py::FeudalManager • worker.py::FeudalWorker • trainer.py::_env_step")
s.card(44,150,192,115,"GLOBAL STATE", "Xₜ: joint information",MUTED,WHITE,19)
s.card(279,150,269,115,"CENTRAL MANAGER", "θ → latents sₜ,ᵢ\nand goals gₜ,ᵢ",PURPLE,LAVENDER,22)
s.card(594,150,324,115,"SHARED WORKER", "πφ(aₜ,ᵢ | oₜ,ᵢ, uₜ,ᵢ)",TEAL,PALE,22)
s.line(239,208,272,208,PURPLE,2.5,True)
s.line(551,208,587,208,PURPLE,2.5,True)
s.text("pool goals",554,275,115,13,PURPLE)
s.card(44,318,504,104,"CENTRALIZED CRITICS", "Predict extrinsic, manager and optional intrinsic returns.",MUTED,WHITE,19)
s.card(594,318,324,104,"LOCAL OBSERVATION", "oₜ,ᵢ + goal conditioning uₜ,ᵢ",TEAL,WHITE,19)
s.line(756,313,756,273,TEAL,2.5,True)

# 03 — Time semantics
s=Slide("01 / base algorithm", "A new goal every step; progress over c steps", "The base horizon is a pooling and scoring window, not a goal-holding interval.",
"The default c is 10. The manager emits a normalized goal at every step. The worker pools up to the last c goals from the current episode and normally normalizes this pooled vector. K_t contains offsets 0 through c-1 that exist inside the current episode and rollout. The manager later compares the change from s_t to s_(t+c) with the goal issued at t. Goals are unit directions and do not prescribe distance. U(v)=v/(||v||+epsilon_g), epsilon_g=1e-6, so zero remains zero. The physical waypoint mode will replace this rolling channel with a held destination.",
"Sources: manager.py::goal_channel / transition_cosine • worker.py::FeudalWorker")
s.eq(r"g_{t,i}=U(M_{\theta,i}(s_t)),\qquad u_{t,i}=U\!\left(\sum_{k\in K_t}g_{t-k,i}\right)",48,133,870,62)
for j,label in enumerate(["t − c + 1","…","t − 1","t","…","t + c"]):
    xx=95+j*151
    s.circle(xx,274,6,PURPLE if j<4 else TEAL)
    s.text(label,xx-45,291,90,18,align="center")
    if j<5:s.line(xx+8,274,xx+143,274,LINE,3)
s.line(95,239,548,239,PURPLE,3)
s.text("Recent goals condition this action",86,208,510,18,PURPLE,True)
s.line(548,348,850,348,TEAL,3,True)
s.text("Future displacement scores gₜ",547,364,366,18,TEAL,True)
s.text("U: stabilized unit normalization   •   i: agent   •   c: horizon   •   Kₜ: valid recent offsets",48,424,862,16,MUTED)

# 04 — no intrinsic
s=Slide("01 / base algorithm", "Without intrinsic reward: both pursue the task", "α = 0 removes goal-following reward; goals still condition the worker.",
"For the no-intrinsic baseline, the worker PPO advantage is the normalized extrinsic advantage only. The manager still learns through its task-weighted transition-alignment objective. Goal conditioning is still active, so this is not the same as zero_goal. There is no intrinsic critic at configured alpha=0. Extrinsic reward usually means the team reward, but the environment can supply per-agent rewards; both the worker and manager consume that supplied extrinsic learning signal. Their separate critics may learn different baselines even with equal discounts.",
"Sources: mappo.py::ppo_update / manager_update • conf/model/feudal.yaml")
s.card(44,144,420,146,"WORKER: ACTION QUALITY", "Reward actions that improve environment return.",TEAL,PALE,24)
s.card(498,144,420,146,"MANAGER: GOAL QUALITY", "Reward goals aligned with useful state changes.",PURPLE,LAVENDER,24)
s.eq(r"A^W_{t,i}=\widehat A^E_{t,i}",66,314,370,58,31,TEAL)
s.eq(r"\mathcal L_M=-\langle\widehat A^M D\rangle_M",520,314,365,58,29,PURPLE)
s.text("E: worker extrinsic stream\nW: advantage used by the worker actor",62,389,385,17,MUTED)
s.text("M: manager extrinsic stream\nD: alignment with c-step displacement",516,389,385,17,MUTED)

# 05 — intrinsic reward
s=Slide("01 / base algorithm", "Intrinsic reward scores the action’s successor", "The reward endpoint is the real post-action state, read before any reset.",
"This is the reward actually wired into trainer._apply_intrinsic_reward. s_t^+ is the latent of the successor produced by action a_t, not a shifted reset observation. At k=0, the reward measures that action's immediate displacement against the current goal. Other terms score cumulative displacement against recent goals. K_t includes only available offsets within the same episode; early steps divide by the number of valid terms, not always by c. State and goal arguments are detached. The legacy helper worker_intrinsic_reward ends at the pre-action state s_t and remains only for diagnostics.",
"Sources: manager.py::worker_intrinsic_reward_aligned • trainer.py::_apply_intrinsic_reward")
s.rect(44,138,874,120,WHITE,LINE,8)
s.eq(r"r^I_{t,i}=\frac{1}{|K_t|}\sum_{k\in K_t}\operatorname{cos}_{\varepsilon}\!\left(s^+_{t,i}-s_{t-k,i},\ g_{t-k,i}\right)",64,163,837,75,30)
s.card(44,285,272,155,"WHAT MOVED?", "s⁺ₜ,ᵢ: the actual successor representation.",TEAL,PALE,20)
s.card(345,285,272,155,"WHAT WAS ASKED?", "gₜ₋ₖ,ᵢ: a recent goal; k = 0 includes this step.",PURPLE,LAVENDER,20)
s.card(646,285,272,155,"HOW IS IT SCORED?", "Cosine alignment, averaged over valid history Kₜ.",AMBER,GOLD,20)

# 06 — mix advantages
s=Slide("01 / base algorithm", "Combine normalized advantages, not raw rewards", "α controls the relative standardized signal; it is not an exact gradient percentage.",
"Each reward stream has its own critic and GAE. Normalize each advantage separately over rollout time, per environment and agent, then combine. This prevents raw reward magnitudes from deciding the mixture by accident. The PPO surrogate uses this mixed advantage. Alpha is a coefficient on standardized advantages, not a guaranteed fraction of the resulting gradient norm; policy score gradients, correlations, clipping and Adam matter. The intrinsic critic is trained against its own unscaled return. Current n001/n01/n05 presets use 0.01/0.1/0.5 and constant alpha. Optional linear annealing multiplies alpha_0 by max(1-p,0), where p is training progress.",
"Sources: mappo.py::ppo_update / _annealed_alpha • conf/model/feudal*n*.yaml")
s.card(44,141,405,116,"EXTRINSIC STREAM", "rᴱ → Vᴱ → GAE → normalized Âᴱ",TEAL,PALE,21)
s.card(513,141,405,116,"INTRINSIC STREAM", "rᴵ → Vᴵ → GAE → normalized Âᴵ",AMBER,GOLD,21)
s.eq(r"A^W_{t,i}=\widehat A^E_{t,i}+\alpha\widehat A^I_{t,i}",196,290,660,76,36)
s.table(["PRESET SUFFIX","INTRINSIC WEIGHT","CURRENT SCHEDULE"],[["_n001 / _n01 / _n05","0.01 / 0.1 / 0.5","Constant (none)"]],[285,250,341],y=377,row_h=53,size=19)

# 07 — GAE
s=Slide("02 / loss functions", "GAE turns rewards into credit over time", "Each stream gets its own value prediction, advantage and fixed return target.",
"q ranges over E (worker extrinsic), I (worker intrinsic), M (manager extrinsic). Agent and environment indices are omitted. v_t^q is the saved rollout value prediction. r-tilde includes a time-limit bootstrap correction described in the appendix; d_t is true for termination or truncation. gamma_q discounts future rewards, and lambda controls the GAE trace. Start the reverse recursion with zero beyond the collected rollout; the final value still bootstraps the final TD error. R=A+v uses the raw advantage and is the critic target. Normalize A over time separately per environment/output using the unbiased sample standard deviation and epsilon_A=1e-8. Manager GAE is per-step, not a c-step-return construction.",
"Source: mappo.py::compute_gae / ppo_update / manager_update")
s.eq(r"\delta_t^q=\widetilde r_t^q+\gamma_q(1-d_t)v_{t+1}^q-v_t^q",48,132,860,54,29)
s.eq(r"A_t^q=\delta_t^q+\gamma_q\lambda(1-d_t)A_{t+1}^q,\qquad R_t^q=A_t^q+v_t^q",48,204,860,58,28)
s.eq(r"\widehat A_t^q=\frac{A_t^q-\operatorname{mean}_t(A^q)}{\operatorname{std}_t(A^q)+\varepsilon_A}",48,287,545,78,27)
s.text("q = E, I, M\nδ: TD error; A: advantage\nR: value target; v: saved value",610,284,302,19,MUTED)
s.text("d: episode boundary   •   γ: discount   •   λ: trace decay   •   εA = 10⁻⁸",48,409,862,17,MUTED)

# 08 — PPO
s=Slide("02 / loss functions", "The worker uses clipped PPO plus entropy", "Re-evaluate the stored action under the same stored goal conditioning.",
"rho is the current action probability divided by the rollout action probability. The numerator and denominator use the same local observation, stored pooled goal and legal-action mask. The old log probability is saved during collection. A^W is the extrinsic-only, mixed or intrinsic-only advantage depending on the variant. The minimum implements PPO's clipped surrogate; epsilon_P=0.2 is the default. H is the entropy of the action distribution and beta=0.01 by default. Angle brackets with subscript W denote the mean over active worker samples. Advantages, actions and stored goals are fixed data during the update. The worker and critics use separate optimizers.",
"Source: mappo.py::ppo_update::_surrogate")
s.eq(r"\rho_{t,i}=\frac{\pi_\phi(a_{t,i}\mid o_{t,i},u_{t,i})}{\pi_{\phi_{\rm old}}(a_{t,i}\mid o_{t,i},u_{t,i})}",48,133,862,81,29)
s.rect(44,236,874,112,WHITE,LINE,8)
s.eq(r"\mathcal L_W=-\left\langle\min\!\left(\rho A^W,\ \operatorname{clip}(\rho,1-\epsilon_P,1+\epsilon_P)A^W\right)\right\rangle_W",61,249,835,55,25)
s.eq(r"\phantom{\mathcal L_W=} -\beta\langle\mathcal H(\pi_\phi)\rangle_W",62,297,830,39,25)
s.text("φ / φold: current / rollout worker parameters\nεP = 0.2: clipping range   •   β = 0.01: entropy weight\n⟨·⟩W: average over active decisions   •   H: action entropy",48,371,850,18,MUTED)

# 09 — manager
s=Slide("02 / loss functions", "The manager reinforces useful transitions", "Task advantage weights alignment; gradients flow through the goal, not the measured change.",
"D is the stabilized cosine between the observed c-step displacement and the manager's goal. sg means stop_gradient. The loss is negative normalized extrinsic manager advantage times D, averaged over active samples with a full horizon inside one episode. If task advantage is positive, the gradient encourages alignment with this displacement. If negative, it discourages alignment. The manager has no goal likelihood ratio, PPO clipping or entropy term. Recompute goals differentiably from the stored ordered states. The state encoder is still trainable through the goal branch because the manager core consumes s. Detaching only the target branch does not make the entire encoder frozen, nor does it guarantee freedom from all collapse mechanisms.",
"Sources: manager.py::transition_cosine • mappo.py::manager_update")
s.eq(r"D_{t,i}=\operatorname{cos}_{\varepsilon}\!\left(\operatorname{sg}[s_{t+c,i}-s_{t,i}],\ g_{t,i}(\theta)\right)",48,132,862,66,30)
s.eq(r"\mathcal L_M=-\left\langle\operatorname{sg}[\widehat A^M_{t,i}]\,D_{t,i}\right\rangle_M",48,218,862,67,33,PURPLE)
s.card(44,321,421,120,"FIXED TARGET", "Observed displacement and advantage are treated as data.",MUTED,WHITE,20)
s.card(497,321,421,120,"TRAINABLE GOAL", "θ receives gradients through goal generation, including its encoder.",PURPLE,LAVENDER,20)

# 10 — critics
s=Slide("02 / loss functions", "Three critics, three separate targets", "Value losses use raw return targets; only policy advantages are standardized.",
"V^E and V^I always have one head per agent; the manager value is scalar for team reward and per-agent if the environment supplies per-agent rewards. All critics read the centralized state X. psi_q denotes the parameters of critic q. The expectation or bracket is an empirical mean of squared errors, with inactive per-agent outputs masked. The coefficient c_q is 0.5 by default for all three, though environment configurations can override values. These are plain MSE regressions without value clipping or a one-half prefactor. V^I is built only when the intrinsic stream is enabled. In intrinsic-only mode V^E still trains, even though its advantage is absent from the actor objective.",
"Sources: mappo.py::create_train_state / ppo_update / manager_update")
s.eq(r"\mathcal L_{V^q}=c_q\left\langle\left(V^q_{\psi_q}(X_t)-R_t^q\right)^2\right\rangle_q",48,129,862,74,32)
s.table(["CRITIC","WHAT IT PREDICTS","OUTPUT / DISCOUNT"],[
    ["Vᴱ: worker extrinsic","Environment return","Per agent / γ"],
    ["Vᴵ: worker intrinsic","Goal-following return","Per agent / γ"],
    ["Vᴹ: manager","Environment return","Team or per agent / γM"],
],[227,313,336],y=225,row_h=54,size=19)
s.text("ψq: critic parameters   •   cq: value-loss weight (default 0.5)   •   X: global state",48,427,867,16,MUTED)

# 11 — update sequence
s=Slide("02 / updates", "Collect once; update worker, then manager", "Separate optimizers keep each objective explicit; the shared-encoder variant adds one exception.",
"Collection freezes all networks and saves action log probabilities, conditioning, critic predictions, rewards, states and boundaries. Post-processing computes intrinsic rewards when enabled. Worker PPO uses shuffled timestep-centric minibatches for six epochs by default; actual minibatch count follows implementation dimensions, so do not infer 48 optimizer steps from the configured n_minibatches alone. During PPO manager parameters do not move. Manager GAE uses saved rollout values. Its critic fits fixed targets for eight full-batch passes, and its actor normally takes one full-batch transition-gradient step. Recurrent goals are reconstructed by replaying the episode-reset logic. Adam uses global gradient-norm clipping (default 0.5), with default learning rates 3e-4. Repeating manager policy passes lacks an importance correction.",
"Sources: trainer.py::update_fn • mappo.py::ppo_update / manager_update")
steps=[("1","COLLECT","Freeze networks;\nsave rollout data.",MUTED),("2","WORKER PPO","6 epochs; actor\nand worker critics.",TEAL),("3","MANAGER VALUE","8 full-batch passes\non fixed targets.",PURPLE),("4","MANAGER GOALS","1 ordered full-batch\ntransition update.",PURPLE)]
for j,(num,label,body,col) in enumerate(steps):
    x=44+j*225
    s.circle(x+23,164,20,col)
    s.text(num,x+8,151,30,23,WHITE,True,align="center")
    s.card(x,209,198,174,label,body,col,WHITE,20)
    if j<3:s.line(x+202,295,x+219,295,col,2,True)
s.text("Adam learning rates: 3 × 10⁻⁴   •   Global gradient-norm clipping: 0.5",48,421,867,18,MUTED)

