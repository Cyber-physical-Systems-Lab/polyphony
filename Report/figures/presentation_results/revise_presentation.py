#!/usr/bin/env python3
"""Build a thesis-focused, presentation-ready revision of the supplied deck.

The script connects to a running headless LibreOffice instance on port 2002,
creates an editable Impress document, and exports it as PPTX.  All claims and
figures are sourced from the thesis and its checked-in presentation assets.
"""

from pathlib import Path
import math
import uno
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
FIG = ROOT / "Report" / "figures"
ASSET = FIG / "presentation_results"
OUT = ASSET / "RohitThesisPresentation_revised.pptx"
LAYOUT_IMG = ASSET / "scenario_layouts_generated.png"

SW, SH = 33867, 19050  # 16:9, units are 1/100 mm
NAVY = 0x102B3F
TEAL = 0x1F7188
ORANGE = 0xF28C28
BLUE = 0x3E78B2
PURPLE = 0x4F4593
GREEN = 0x1F8A70
RED = 0xD9534F
DARK = 0x23313B
MID = 0x5D6B75
LIGHT = 0xEAF2F5
PALE = 0xF7F9FA
WHITE = 0xFFFFFF
GRID = 0xD8E0E5

FILL_NONE = uno.Enum("com.sun.star.drawing.FillStyle", "NONE")
FILL_SOLID = uno.Enum("com.sun.star.drawing.FillStyle", "SOLID")
LINE_NONE = uno.Enum("com.sun.star.drawing.LineStyle", "NONE")
LINE_SOLID = uno.Enum("com.sun.star.drawing.LineStyle", "SOLID")


def point(x, y):
    p = uno.createUnoStruct("com.sun.star.awt.Point")
    p.X, p.Y = int(x), int(y)
    return p


def usize(w, h):
    s = uno.createUnoStruct("com.sun.star.awt.Size")
    s.Width, s.Height = int(w), int(h)
    return s


def prop(name, value):
    p = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
    p.Name = name
    p.Value = value
    return p


def connect():
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local
    )
    ctx = resolver.resolve(
        "uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
    )
    smgr = ctx.ServiceManager
    desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    return ctx, smgr, desktop


def set_if(obj, name, value):
    try:
        setattr(obj, name, value)
    except Exception:
        pass


def add_text(doc, page, x, y, w, h, text, size=20, color=DARK, bold=False,
             font="Lato", align=0, valign=0, margin=120, name=""):
    s = doc.createInstance("com.sun.star.drawing.TextShape")
    s.Position = point(x, y)
    s.Size = usize(w, h)
    page.add(s)
    s.String = text
    set_if(s, "CharFontName", font)
    set_if(s, "CharHeight", float(size))
    set_if(s, "CharColor", color)
    set_if(s, "CharWeight", 150.0 if bold else 100.0)
    set_if(s, "ParaAdjust", align)
    set_if(s, "TextVerticalAdjust", valign)
    set_if(s, "TextLeftDistance", margin)
    set_if(s, "TextRightDistance", margin)
    set_if(s, "TextUpperDistance", margin)
    set_if(s, "TextLowerDistance", margin)
    set_if(s, "FillStyle", FILL_NONE)
    set_if(s, "LineStyle", LINE_NONE)
    if name:
        s.Name = name
    return s


def add_box(doc, page, x, y, w, h, fill=WHITE, line=GRID, radius=False,
            text="", size=18, color=DARK, bold=False, align=3, valign=2,
            line_width=35):
    service = "com.sun.star.drawing.RectangleShape"
    s = doc.createInstance(service)
    s.Position = point(x, y)
    s.Size = usize(w, h)
    page.add(s)
    s.FillStyle = FILL_SOLID
    s.FillColor = fill
    s.LineStyle = LINE_SOLID if line is not None else LINE_NONE
    if line is not None:
        s.LineColor = line
        s.LineWidth = line_width
    if radius:
        set_if(s, "CornerRadius", 220)
    if text:
        s.String = text
        set_if(s, "CharFontName", "Lato")
        set_if(s, "CharHeight", float(size))
        set_if(s, "CharColor", color)
        set_if(s, "CharWeight", 150.0 if bold else 100.0)
        set_if(s, "ParaAdjust", align)
        set_if(s, "TextVerticalAdjust", valign)
        set_if(s, "TextLeftDistance", 180)
        set_if(s, "TextRightDistance", 180)
        set_if(s, "TextUpperDistance", 120)
        set_if(s, "TextLowerDistance", 120)
    return s


def add_line(doc, page, x1, y1, x2, y2, color=TEAL, width=70, arrow=True):
    s = doc.createInstance("com.sun.star.drawing.LineShape")
    s.Position = point(x1, y1)
    s.Size = usize(x2 - x1, y2 - y1)
    s.LineColor = color
    s.LineWidth = width
    if arrow:
        set_if(s, "LineEndName", "Arrow")
        set_if(s, "LineEndWidth", 250)
    page.add(s)
    return s


def add_image(doc, page, path, x, y, w, h, crop=False):
    shape = doc.createInstance("com.sun.star.drawing.GraphicObjectShape")
    shape.Position = point(x, y)
    shape.Size = usize(w, h)
    shape.GraphicURL = uno.systemPathToFileUrl(str(path))
    page.add(shape)
    return shape


def clear_page(page):
    while page.getCount():
        page.remove(page.getByIndex(0))


def set_notes(page, notes):
    notes_page = page.getNotesPage()
    target = None
    for i in range(notes_page.getCount()):
        s = notes_page.getByIndex(i)
        if "NotesShape" in getattr(s, "ShapeType", ""):
            target = s
            break
    if target is not None:
        target.String = notes
        set_if(target, "CharFontName", "Lato")
        set_if(target, "CharHeight", 12.0)


def base_slide(doc, pages, title, section="", notes=""):
    if pages.getCount() == 1 and pages.getByIndex(0).getCount() == 0:
        page = pages.getByIndex(0)
    else:
        page = pages.insertNewByIndex(pages.getCount())
    page.Width, page.Height = SW, SH
    add_text(doc, page, 1500, 700, 30000, 1300, title, 29, NAVY, True)
    add_box(doc, page, 1500, 17650, 30800, 20, fill=TEAL, line=None)
    add_text(doc, page, 1500, 17820, 20000, 450,
             "Master’s thesis · Uppsala University", 9, MID)
    if section:
        add_text(doc, page, 26700, 17820, 5600, 450, section.upper(), 8, TEAL,
                 True, align=1)
    set_notes(page, notes)
    return page


def title_slide(doc, pages):
    page = pages.getByIndex(0)
    clear_page(page)
    page.Width, page.Height = SW, SH
    add_box(doc, page, 0, 0, SW, SH, fill=PALE, line=None)
    add_box(doc, page, 0, 0, 950, SH, fill=TEAL, line=None)
    add_text(doc, page, 2500, 3350, 27500, 2800,
             "Heterogeneous Multi-Robot Collaboration\nUsing Language Models",
             34, NAVY, True)
    add_text(doc, page, 2550, 7100, 18000, 1700,
             "Can locally deployable language models support\ncollaborative high-level decisions in a warehouse simulator?",
             21, MID)
    add_box(doc, page, 2300, 10600, 12500, 2500, fill=WHITE, line=GRID, radius=True)
    add_text(doc, page, 2800, 11000, 11300, 1700,
             "Rohit Joseph Mamutil\nMaster’s Thesis · Embedded Systems\nUppsala University",
             17, DARK, False)
    add_box(doc, page, 23800, 10600, 6200, 2500, fill=NAVY, line=NAVY, radius=True,
            text="LLM\n→\nteam decision", size=20, color=WHITE, bold=True)
    add_text(doc, page, 2550, 17450, 27500, 500,
             "Battery-TA-RWARE · local inference · controlled simulation", 11, TEAL, True)
    set_notes(page,
        "Good morning. I am Rohit Joseph Mamutil, and this thesis asks a focused question: "
        "can locally deployable language models support high-level decisions when robots with "
        "different roles must work together? I study that question in a controlled warehouse "
        "simulator. The language model chooses task-level actions; the simulator retains navigation "
        "and execution. I will first define the collaboration problem, then show the experimental "
        "system and scenarios, and finally present the three main comparisons and their limits.")


def generate_layout_image():
    W, H = 1800, 850
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Bold.ttf", 34)
        font = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Regular.ttf", 25)
        small = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Regular.ttf", 20)
    except Exception:
        title_font = font = small = None

    def draw_layout(x0, y0, rows, cols, label, dims):
        gh = 15 if rows == 1 else 25
        gw = 14 if cols == 3 else 22
        maxw, maxh = 480, 620
        cell = min(maxw / gw, maxh / gh)
        ox = x0 + (maxw - gw * cell) / 2
        oy = y0 + 75 + (maxh - gh * cell) / 2
        d.text((x0, y0), label, font=title_font, fill=(16, 43, 63))
        highway_x = {i + j for i in range(0, gw, 4) for j in range(2)}
        highway_y = {i + j for i in range(0, gh, 10) for j in range(2)}
        for yy in range(gh):
            for xx in range(gw):
                x1, y1 = ox + xx * cell, oy + yy * cell
                is_bottom = yy >= gh - 3
                if yy == 0:
                    fill = (242, 140, 40)  # chargers
                elif yy == gh - 1 and xx not in highway_x:
                    fill = (31, 138, 112)  # goals
                elif xx not in highway_x and yy not in highway_y and not is_bottom:
                    fill = (79, 69, 147)  # shelves
                else:
                    fill = (244, 247, 249)
                d.rectangle((x1, y1, x1 + cell - 1, y1 + cell - 1), fill=fill,
                            outline=(216, 224, 229))
        d.text((x0, y0 + 735), dims, font=font, fill=(93, 107, 117))

    draw_layout(80, 35, 1, 3, "TINY", "15 × 14 grid · 1 shelf row")
    draw_layout(660, 35, 2, 3, "SMALL", "25 × 14 grid · 2 shelf rows")
    draw_layout(1240, 35, 2, 5, "MEDIUM", "25 × 22 grid · 5 shelf columns")
    # Legend
    for i, (label, color) in enumerate([
        ("shelves", (79, 69, 147)), ("goals", (31, 138, 112)),
        ("chargers", (242, 140, 40)), ("travel lanes", (244, 247, 249))]):
        x = 580 + i * 190
        d.rectangle((x, 810, x + 26, 836), fill=color, outline=(150, 160, 166))
        d.text((x + 36, 806), label, font=small, fill=(35, 49, 59))
    im.save(LAYOUT_IMG)


def card(doc, page, x, y, w, h, kicker, title, body, accent):
    add_box(doc, page, x, y, w, h, fill=WHITE, line=GRID, radius=True)
    add_box(doc, page, x, y, 180, h, fill=accent, line=None)
    add_text(doc, page, x + 450, y + 350, w - 700, 500, kicker, 12, accent, True)
    add_text(doc, page, x + 450, y + 1050, w - 700, 1350, title, 22, NAVY, True)
    add_text(doc, page, x + 450, y + 2750, w - 750, h - 3100, body, 16, DARK)


def build_deck(doc):
    pages = doc.getDrawPages()
    clear_page(pages.getByIndex(0))
    title_slide(doc, pages)

    p = base_slide(doc, pages, "Why heterogeneous robot collaboration?", "Motivation",
        "A warehouse makes interdependence visible. An AGV can move a shelf, while a Picker "
        "provides support at shelf interaction points. Neither role completes the full cycle alone. "
        "At the same time, robots share paths, shelves, goals, and charging stations. The research "
        "problem is therefore not simply which action is best for one robot, but whether a language "
        "model can choose high-level actions that let complementary roles make progress together.")
    add_image(doc, p, FIG / "KivaWarehouse.png", 1550, 2650, 13300, 11700)
    add_text(doc, p, 16500, 3300, 14500, 700,
             "The decision problem becomes interdependent", 23, NAVY, True)
    add_text(doc, p, 16500, 4600, 14200, 5200,
             "• AGVs transport movable shelves\n\n• Pickers support loading and unloading\n\n"
             "• Roles share paths, goals, and chargers\n\n• One robot’s choice creates work for another",
             19, DARK)
    add_box(doc, p, 16500, 11100, 14200, 2500, fill=LIGHT, line=TEAL, radius=True,
            text="Question: can an LLM choose useful task-level actions for the team?",
            size=21, color=NAVY, bold=True)

    p = base_slide(doc, pages, "The collaboration cycle", "Problem",
        "This cycle defines collaboration in the simulator. An AGV first travels to a requested "
        "shelf. A Picker must meet it at the same shelf action so loading can occur. The AGV then "
        "delivers the shelf to a goal; no Picker is needed there. Finally, the AGV returns the shelf "
        "to an empty storage location, where Picker support is required again. The key experimental "
        "dependency is the two support events: pickup and return.")
    labels = [
        ("1", "Requested shelf", "AGV selects and travels", BLUE),
        ("2", "Load together", "Picker meets the AGV", ORANGE),
        ("3", "Deliver", "AGV moves shelf to goal", GREEN),
        ("4", "Return shelf", "AGV selects empty location", PURPLE),
        ("5", "Unload together", "Picker support completes cycle", ORANGE),
    ]
    xs = [1200, 7600, 14000, 20400, 26800]
    for i, (num, title, body, col) in enumerate(labels):
        add_box(doc, p, xs[i], 5000, 5100, 5200, fill=WHITE, line=col, radius=True)
        add_box(doc, p, xs[i] + 1850, 4200, 1400, 1400, fill=col, line=col, radius=True,
                text=num, size=24, color=WHITE, bold=True)
        add_text(doc, p, xs[i] + 350, 6300, 4400, 700, title, 20, NAVY, True, align=3)
        add_text(doc, p, xs[i] + 450, 7600, 4200, 1300, body, 16, DARK, align=3)
        if i < len(labels) - 1:
            add_line(doc, p, xs[i] + 5200, 7550, xs[i + 1] - 100, 7550, MID, 55)
    add_text(doc, p, 5900, 11900, 21800, 1500,
             "Collaboration is required at shelf pickup and shelf return—not at the delivery goal.",
             22, TEAL, True, align=3)

    p = base_slide(doc, pages, "Research focus and boundary", "Questions",
        "The thesis evaluates LLMs as high-level decision components, not as end-to-end robot "
        "controllers. The model sees a symbolic snapshot and candidate actions, then selects what "
        "an agent should pursue. The simulator handles how that action is executed. The main question "
        "is whether prompted LLMs can produce executable decisions that lead to successful "
        "heterogeneous collaboration in this constrained setting.")
    add_box(doc, p, 1800, 3400, 30000, 2600, fill=NAVY, line=NAVY, radius=True,
            text="Can prompted LLMs support executable high-level decisions\nfor heterogeneous robot collaboration?",
            size=26, color=WHITE, bold=True)
    card(doc, p, 2400, 7500, 8600, 5500, "IN SCOPE", "High-level decisions",
         "Choose a shelf, goal, return location, charger, or no-op.", TEAL)
    card(doc, p, 12600, 7500, 8600, 5500, "IN SCOPE", "Collaborative outcome",
         "Successful shelf deliveries across heterogeneous AGV–Picker teams.", ORANGE)
    card(doc, p, 22800, 7500, 8600, 5500, "OUT OF SCOPE", "Low-level robotics",
         "No motor control, trajectory generation, or raw sensor perception.", MID)

    p = base_slide(doc, pages, "Three design questions", "Questions",
        "The broad question becomes three controlled comparisons. RQ1 varies the selected local "
        "model and asks whether performance rises consistently with nominal parameter size. RQ2 "
        "compares one joint planning call with separate per-agent decisions informed by shared context. "
        "RQ3 compares the complete natural-language and JSON prompt interfaces. These are system-design "
        "questions about how an LLM is used, not only which model is used.")
    card(doc, p, 1400, 3400, 9600, 9300, "RQ1 · MODEL", "Which model?",
         "Six locally deployable models\n1B–14B nominal scale\n\nDoes task performance increase consistently with size?", BLUE)
    card(doc, p, 12100, 3400, 9600, 9300, "RQ2 · ARCHITECTURE", "Who plans?",
         "Centralized joint planning\nversus\nshared-context per-agent planning", ORANGE)
    card(doc, p, 22800, 3400, 9600, 9300, "RQ3 · PROMPT", "How is state presented?",
         "Descriptive natural language\nversus\nstructured JSON configuration", GREEN)

    p = base_slide(doc, pages, "Battery-TA-RWARE environment", "Environment",
        "Battery-TA-RWARE is a discrete-time grid warehouse derived from TA-RWARE. Purple cells are "
        "shelf positions, green cells mark requested shelves, the lower edge contains delivery goals, "
        "and the top edge contains charging stations. AGVs and Pickers move through shared lanes. "
        "The environment supplies a global symbolic state for these experiments and handles A-star "
        "paths, collision resolution, battery updates, and shelf mechanics.")
    add_image(doc, p, FIG / "tarware_explanation.png", 1200, 3000, 21000, 13200)
    card(doc, p, 23500, 3300, 8200, 3100, "ROLES", "AGV + Picker",
         "Transport and support are complementary capabilities.", TEAL)
    card(doc, p, 23500, 7000, 8200, 3100, "RESOURCES", "Shared workspace",
         "Shelves, goals, travel lanes, and charging stations.", ORANGE)
    card(doc, p, 23500, 10700, 8200, 4100, "EXECUTION", "Simulator-owned",
         "A* navigation, collisions, battery dynamics, loading and unloading.", BLUE)

    p = base_slide(doc, pages, "Three warehouse layouts", "Scenarios",
        "The scenario set varies spatial scale as well as team composition. The tiny layout is a "
        "15 by 14 grid with one shelf row. The small layout extends this to 25 by 14 with two shelf "
        "rows. The medium layout is also 25 cells high but widens to 22 cells and five shelf columns. "
        "Larger layouts increase path length, shelf positions, and the number of simultaneous decisions.")
    add_image(doc, p, LAYOUT_IMG, 1100, 2600, 31600, 14900)

    p = base_slide(doc, pages, "Six scenarios: scale × role balance", "Scenarios",
        "The six scenarios are not six arbitrary maps. They vary two sources of difficulty. Spatial "
        "scale changes from tiny to small to medium. Role balance changes from balanced teams to "
        "AGV-heavy or Picker-heavy teams. AGV-heavy cases make Picker support scarce; the Picker-heavy "
        "case tests whether surplus support capacity is used productively. This design lets the results "
        "be compared across distinct collaboration demands.")
    add_text(doc, p, 2500, 2900, 5600, 650, "LAYOUT", 13, MID, True)
    add_text(doc, p, 8800, 2900, 6600, 650, "BALANCED", 13, BLUE, True, align=3)
    add_text(doc, p, 16400, 2900, 6600, 650, "AGV-HEAVY", 13, ORANGE, True, align=3)
    add_text(doc, p, 24000, 2900, 6600, 650, "PICKER-HEAVY", 13, GREEN, True, align=3)
    rows = [
        ("TINY", "1 AGV + 1 Picker\nMinimal collaboration loop", "—", "—"),
        ("SMALL", "2 AGVs + 2 Pickers\nMore concurrency", "4 AGVs + 2 Pickers\nSupport bottleneck", "2 AGVs + 4 Pickers\nSurplus support"),
        ("MEDIUM", "4 AGVs + 4 Pickers\nLonger paths + more decisions", "6 AGVs + 2 Pickers\nStrong bottleneck", "—"),
    ]
    for r, row in enumerate(rows):
        y = 4000 + r * 3800
        add_box(doc, p, 1800, y, 6000, 3000, fill=NAVY, line=NAVY, radius=True,
                text=row[0], size=22, color=WHITE, bold=True)
        for c in range(3):
            txt = row[1 + c]
            fill = WHITE if txt != "—" else PALE
            col = [BLUE, ORANGE, GREEN][c]
            add_box(doc, p, 8500 + c * 7600, y, 6900, 3000, fill=fill,
                    line=col if txt != "—" else GRID, radius=True,
                    text=txt, size=17, color=DARK if txt != "—" else MID,
                    bold=txt != "—")

    p = base_slide(doc, pages, "What the LLM controls", "Implementation",
        "The model operates only at the task-assignment level. Its output is an integer action ID and "
        "a short persistence duration. Depending on role and state, that ID may refer to a requested "
        "shelf, a goal, an empty return location, a charging station, or no-op. The simulator translates "
        "that target into motion and physical interaction. This boundary keeps the experiment focused "
        "on high-level decisions rather than navigation quality.")
    add_box(doc, p, 1800, 4200, 12500, 8200, fill=LIGHT, line=TEAL, radius=True)
    add_text(doc, p, 2600, 5000, 10900, 700, "LANGUAGE MODEL", 14, TEAL, True, align=3)
    add_text(doc, p, 2600, 6300, 10900, 1300, "WHAT should this agent pursue?", 25, NAVY, True, align=3)
    add_text(doc, p, 3200, 8500, 9800, 2400,
             "Shelf · Goal · Empty return\nCharging · No-op", 20, DARK, align=3)
    add_line(doc, p, 14500, 8300, 18400, 8300, ORANGE, 100)
    add_box(doc, p, 18800, 4200, 12500, 8200, fill=PALE, line=BLUE, radius=True)
    add_text(doc, p, 19600, 5000, 10900, 700, "SIMULATOR", 14, BLUE, True, align=3)
    add_text(doc, p, 19600, 6300, 10900, 1300, "HOW is the action executed?", 25, NAVY, True, align=3)
    add_text(doc, p, 20200, 8500, 9800, 2400,
             "A* path · Collisions · Battery\nLoad / unload mechanics", 20, DARK, align=3)

    p = base_slide(doc, pages, "Implementation: the supervisory decision loop", "Implementation",
        "At a planning event, the controller reads the current simulator state and builds a symbolic "
        "snapshot. It then computes currently allowed actions and shapes a compact role-specific "
        "candidate set. A prompt is sent to a locally hosted Ollama model. The response is parsed, "
        "checked against the current action set, and either accepted or replaced by a fallback. The "
        "simulator executes the macro action, while selective replanning is triggered by completion, "
        "expiry, invalidation, or coordination needs.")
    stages = [
        ("1", "State", "agents · shelves\nbattery · support", BLUE),
        ("2", "Candidates", "role + state\nallowed actions", TEAL),
        ("3", "Prompt", "objective + context\noutput contract", PURPLE),
        ("4", "Local LLM", "Ollama\ntemperature 0.1", ORANGE),
        ("5", "Parse + check", "action ID + steps\nfallback on failure", RED),
        ("6", "Execute", "macro action\nselective replanning", GREEN),
    ]
    for i, (n, t, b, col) in enumerate(stages):
        x = 1050 + i * 5350
        add_box(doc, p, x, 4800, 4550, 6000, fill=WHITE, line=col, radius=True)
        add_box(doc, p, x + 150, 4950, 850, 850, fill=col, line=col, radius=True,
                text=n, size=16, color=WHITE, bold=True)
        add_text(doc, p, x + 350, 6500, 3850, 650, t, 20, NAVY, True, align=3)
        add_text(doc, p, x + 350, 8000, 3850, 1400, b, 15, DARK, align=3)
        if i < len(stages) - 1:
            add_line(doc, p, x + 4600, 7800, x + 5250, 7800, MID, 50)
    add_box(doc, p, 4300, 12100, 25200, 1900, fill=LIGHT, line=TEAL, radius=True,
            text="The observed behaviour belongs to this complete loop—not to the model in isolation.",
            size=21, color=NAVY, bold=True)

    p = base_slide(doc, pages, "Prompt interface: symbolic state → action ID", "Implementation",
        "The prompt does not expose the entire raw grid. It provides task semantics, the current "
        "agent state, relevant shared context, candidate actions, warnings, and a precise output "
        "contract. This example from the thesis shows the same choice in both prompt configurations. "
        "The experiment therefore compares complete interfaces: descriptive text and text parsing "
        "versus structured JSON and structured-output mode.")
    add_box(doc, p, 1500, 3000, 14600, 11900, fill=PALE, line=GRID, radius=True)
    add_text(doc, p, 2200, 3500, 13200, 600, "REPRESENTATIVE INPUT", 13, TEAL, True)
    add_text(doc, p, 2200, 4650, 13000, 8100,
             "Current agent\nagent_0 · AGV · position (8,3)\nbattery 72 · carrying none\n\nCandidate actions\n0   NOOP       distance 0\n50  SHELF      distance 8\n7   CHARGING   distance 11\n\nShared context\nrequested shelves · support needs\ncharging occupancy · conflict warnings",
             17, DARK, False, font="Liberation Mono")
    add_box(doc, p, 17500, 3000, 14800, 5200, fill=LIGHT, line=TEAL, radius=True)
    add_text(doc, p, 18200, 3500, 13400, 600, "NATURAL-LANGUAGE OUTPUT", 13, TEAL, True)
    add_text(doc, p, 18200, 4700, 13400, 2400,
             "Action: 50\nSteps: 8\nReason: Move to the requested shelf.",
             18, DARK, False, font="Liberation Mono")
    add_box(doc, p, 17500, 9200, 14800, 5700, fill=WHITE, line=PURPLE, radius=True)
    add_text(doc, p, 18200, 9700, 13400, 600, "JSON OUTPUT", 13, PURPLE, True)
    add_text(doc, p, 18200, 11100, 13400, 2100,
             '{"reason":"Move to the requested shelf.",\n "action":50, "steps":8}',
             17, DARK, False, font="Liberation Mono")

    p = base_slide(doc, pages, "Two planning architectures", "Implementation",
        "RQ2 changes who receives each planning problem. Centralized planning sends one global prompt "
        "containing a block for every agent needing a decision, and asks for a joint action list. "
        "Shared-context planning queries agents one at a time. Each query receives the current agent’s "
        "state plus relevant shared information and prior accepted commitments. The state is executed "
        "only after the required decisions are collected.")
    card(doc, p, 1800, 3400, 13700, 10300, "CENTRALIZED", "One joint call",
         "Global snapshot\n+ all replanned agents\n\nOutput: joint action list\n\nFewer model interactions,\nbut a more complex output problem.", BLUE)
    card(doc, p, 18300, 3400, 13700, 10300, "SHARED CONTEXT", "One call per agent",
         "Current-agent state\n+ relevant shared information\n+ accepted commitments\n\nOutput: one action\n\nSimpler decisions, but more calls.", ORANGE)
    add_text(doc, p, 15500, 7600, 2800, 700, "VS", 23, MID, True, align=3)

    p = base_slide(doc, pages, "Models and fixed inference conditions", "Experiment",
        "Six released, locally deployable models were evaluated without task-specific fine-tuning. "
        "They span multiple families, so parameter count is not isolated as a causal variable. All "
        "models used the same fixed hardware platform, an NVIDIA RTX A4000, and were served locally "
        "through Ollama. Prompt templates, state abstraction, parsing, action checks, and metrics were "
        "held constant for each matched comparison.")
    models = [("Llama 3.2", "1B", BLUE), ("Llama 3.2", "3B", BLUE),
              ("Mistral", "7B", TEAL), ("Gemma 3", "12B", GREEN),
              ("Phi-4", "≈14B", PURPLE), ("Qwen 2.5", "14B", ORANGE)]
    for i, (name, size, col) in enumerate(models):
        x = 1400 + (i % 3) * 10500
        y = 3500 + (i // 3) * 4600
        add_box(doc, p, x, y, 9200, 3600, fill=WHITE, line=col, radius=True)
        add_text(doc, p, x + 500, y + 550, 5200, 650, name, 21, NAVY, True)
        add_text(doc, p, x + 6100, y + 600, 2400, 700, size, 22, col, True, align=1)
        add_text(doc, p, x + 500, y + 1900, 7800, 550, "local · unchanged weights", 13, MID)
    add_box(doc, p, 6100, 13200, 21600, 1500, fill=NAVY, line=NAVY, radius=True,
            text="Fixed platform: NVIDIA RTX A4000 · Ollama · temperature 0.1",
            size=18, color=WHITE, bold=True)

    p = base_slide(doc, pages, "Controlled comparison and outcome measures", "Experiment",
        "The experiment varies five factors: model, planning architecture, prompt configuration, "
        "scenario, and objective wording. The main RQ comparisons use matched configuration-level "
        "observations; repeated executions with the same configuration identifiers are averaged before "
        "matching. Mean completed shelf deliveries is the primary outcome. LLM calls and deliveries per "
        "one thousand calls are secondary measures of interaction frequency, not compute cost.")
    add_text(doc, p, 1700, 3100, 14500, 700, "SYSTEMATICALLY VARIED", 13, TEAL, True)
    factors = ["Model", "Architecture", "Prompt", "Scenario", "Objective"]
    for i, f in enumerate(factors):
        add_box(doc, p, 1700, 4300 + i * 1900, 12600, 1350, fill=WHITE,
                line=[BLUE, ORANGE, GREEN, PURPLE, TEAL][i], radius=True,
                text=f, size=19, color=NAVY, bold=True)
    add_text(doc, p, 17800, 3100, 14500, 700, "REPORTED MEASURES", 13, ORANGE, True)
    card(doc, p, 17500, 4300, 14500, 2700, "PRIMARY", "Mean shelf deliveries",
         "How much of the common collaborative task was completed.", GREEN)
    card(doc, p, 17500, 7600, 14500, 2700, "SECONDARY", "Mean LLM calls",
         "Frequency of model interactions during a run.", ORANGE)
    card(doc, p, 17500, 10900, 14500, 2700, "SECONDARY", "Deliveries / 1000 calls",
         "Call-normalised productivity; not latency or compute cost.", BLUE)

    p = base_slide(doc, pages, "Results at a glance", "Results",
        "The three comparisons produce a coherent answer. First, model choice matters, but the ranking "
        "does not rise consistently with nominal size. Second, shared-context planning produced more "
        "deliveries across all six scenarios, while centralized planning used fewer calls. Third, the "
        "complete natural-language prompt configuration outperformed the JSON configuration in all six "
        "scenarios. These findings demonstrate feasibility within the simulator, not superiority over "
        "non-LLM planners.")
    card(doc, p, 1400, 3600, 9600, 8500, "RQ1", "Model choice mattered",
         "Llama 3B: highest mean deliveries\nPhi-4: highest call-normalised value\n\nNominal size alone did not predict performance.", BLUE)
    card(doc, p, 12100, 3600, 9600, 8500, "RQ2", "Shared context delivered more",
         "Higher mean deliveries in all six scenarios and for five of six models.\n\nCentralized planning required fewer calls.", ORANGE)
    card(doc, p, 22800, 3600, 9600, 8500, "RQ3", "Natural language led overall",
         "Higher deliveries and call-normalised values in all six scenarios.\n\nPhi-4 was the model-level exception.", GREEN)
    add_box(doc, p, 6000, 13300, 21800, 1400, fill=NAVY, line=NAVY, radius=True,
            text="Main claim: feasible in Battery-TA-RWARE—not proven better than established planners.",
            size=18, color=WHITE, bold=True)

    p = base_slide(doc, pages, "RQ1 — performance did not scale monotonically", "Results",
        "This balanced comparison uses the same 48 matched configurations for every model: 24 natural-"
        "language and 24 JSON configurations. Llama 3B achieved the highest mean delivery count at "
        "7.21. Phi-4 achieved the highest call-normalised value at 7.86 deliveries per thousand calls. "
        "Larger selected models did not uniformly dominate, so nominal parameter size alone did not "
        "predict collaborative task performance among these model artefacts.")
    add_image(doc, p, ASSET / "rq1_matched_model_performance.png", 1600, 2700, 30600, 12700)

    p = base_slide(doc, pages, "RQ1 — the engineering takeaway", "Results",
        "The result is not that larger models are worse. These are cross-family comparisons involving "
        "different training and post-training choices, so parameter count is not a causal variable. "
        "The useful conclusion is narrower: selecting a planner by nominal size alone would have missed "
        "strong smaller-model performance. Smaller locally deployable models therefore remain viable "
        "candidates, although this study did not directly measure latency, energy, or memory use.")
    card(doc, p, 1800, 3900, 9200, 8200, "OBSERVED", "Rankings changed",
         "Llama 3B led mean deliveries.\nPhi-4 led the call-normalised measure.\nNo monotonic size trend appeared.", BLUE)
    card(doc, p, 12300, 3900, 9200, 8200, "DO NOT CLAIM", "Size caused the result",
         "Models come from different families and differ in training, architecture, and post-training.", RED)
    card(doc, p, 22800, 3900, 9200, 8200, "IMPLICATION", "Benchmark the full setup",
         "Choose the model together with the prompt, architecture, task, and deployment constraints.", GREEN)

    p = base_slide(doc, pages, "RQ2 — shared context improved deliveries", "Results",
        "Across 16 matched configuration-level observations per architecture and scenario, shared-context "
        "planning produced higher mean deliveries in all six scenario groups. The advantage therefore "
        "was not restricted to a single warehouse size or role balance. However, shared-context planning "
        "structurally makes separate per-agent calls, so raw deliveries—not call-normalised efficiency—"
        "is the primary architecture measure.")
    add_image(doc, p, ASSET / "rq2_architecture_by_scenario.png", 1500, 2700, 30800, 12700)

    p = base_slide(doc, pages, "RQ2 — an architecture–model trade-off", "Results",
        "At model level, shared-context planning produced more deliveries for five models. Qwen was the "
        "exception and produced more under centralized planning. A plausible interpretation is that "
        "single-agent outputs simplify each decision, while centralized planning asks the model to "
        "construct a compatible joint action. This mechanism was not isolated experimentally. The safe "
        "conclusion is a trade-off: shared context generally improved delivery, while centralized "
        "planning required fewer model interactions.")
    add_image(doc, p, ASSET / "rq2_architecture_by_model.png", 1400, 2900, 19000, 11800)
    card(doc, p, 21800, 4100, 9800, 3800, "GENERAL PATTERN", "Shared context",
         "Higher delivery performance\n5 of 6 models", ORANGE)
    card(doc, p, 21800, 8800, 9800, 3800, "TRADE-OFF", "Centralized",
         "Fewer model interactions\nQwen reversed the trend", BLUE)

    p = base_slide(doc, pages, "RQ3 — natural language was stronger across scenarios", "Results",
        "For each scenario, 24 matched configurations were compared per prompt condition. The complete "
        "natural-language configuration produced both higher mean deliveries and higher deliveries per "
        "one thousand calls in all six scenarios. The same underlying state variables were available in "
        "both conditions. The result concerns the complete interface, including response constraints and "
        "parsing, rather than input syntax alone.")
    add_image(doc, p, ASSET / "rq3_prompt_by_scenario.png", 1500, 2700, 30800, 12700)

    p = base_slide(doc, pages, "RQ3 — consistent overall, but model-dependent", "Results",
        "Natural language also led within both planning architectures and for five of the six models. "
        "Phi-4 showed a small JSON advantage. This exception matters because it prevents a universal "
        "claim that natural language is inherently better. A reasonable interpretation is that the "
        "descriptive prompt made roles, dependencies, and candidate meanings explicit for most models, "
        "but the experiment did not isolate input representation from output format.")
    add_image(doc, p, ASSET / "rq3_prompt_by_architecture.png", 1300, 3100, 14800, 10400)
    add_image(doc, p, ASSET / "rq3_prompt_by_model.png", 17200, 3100, 14800, 10400)
    add_text(doc, p, 2100, 14100, 29600, 700,
             "Interpretation boundary: compare complete prompt configurations—not JSON syntax alone.",
             18, NAVY, True, align=3)

    p = base_slide(doc, pages, "Supplementary result — objective sensitivity was limited", "Results",
        "Objective wording was an exploratory analysis, not one of the three main research questions. "
        "Four models reduced calls when asked to do so, but Mistral and Gemma increased them. Battery-"
        "focused prompting also produced small, mixed changes across battery indicators and deliveries. "
        "The models were sensitive to wording, but they did not consistently optimize the requested "
        "objective. I therefore treat this as limited objective sensitivity rather than strong adaptation.")
    add_image(doc, p, ASSET / "supplementary_call_objective_response.png", 1500, 3100, 19000, 11200)
    card(doc, p, 21800, 4300, 9800, 3300, "CALL OBJECTIVE", "Mixed response",
         "4 models reduced calls; 2 increased them.", ORANGE)
    card(doc, p, 21800, 8500, 9800, 3900, "BATTERY OBJECTIVE", "Small, mixed changes",
         "No consistent improvement across direct battery outcomes.", TEAL)

    p = base_slide(doc, pages, "Answer to the main question", "Synthesis",
        "Within Battery-TA-RWARE, the answer is yes. Successful shelf deliveries occurred across the "
        "evaluated models, architectures, prompt configurations, and scenarios. That shows that prompted "
        "language models can participate in a constrained high-level decision loop for heterogeneous "
        "robot collaboration. The claim is feasibility within this simulator. Because there is no "
        "non-LLM baseline, it is not a claim of superiority over conventional planning or reinforcement "
        "learning.")
    add_box(doc, p, 2500, 3600, 28800, 3100, fill=NAVY, line=NAVY, radius=True,
            text="YES—prompted LLMs produced executable decisions that completed\ncollaborative shelf-delivery cycles in the tested simulator.",
            size=26, color=WHITE, bold=True)
    add_line(doc, p, 16900, 7000, 16900, 9200, ORANGE, 110)
    card(doc, p, 3000, 9700, 12600, 4200, "SUPPORTED", "Feasibility",
         "Observed across models, architectures, prompts, and scenarios.", GREEN)
    card(doc, p, 18200, 9700, 12600, 4200, "NOT ESTABLISHED", "Superiority",
         "No conventional-planner or reinforcement-learning baseline.", RED)

    p = base_slide(doc, pages, "The contribution is a complete decision system", "Synthesis",
        "The results should not be attributed to the language model alone. The controller decides what "
        "state to expose, which candidate actions to show, how planning is distributed, how outputs are "
        "parsed, and when replanning occurs. The model contributes the high-level selection within that "
        "system. This system-level view explains why prompt configuration and planning architecture had "
        "large effects and is the main engineering lesson of the thesis.")
    stages = [("State abstraction", BLUE), ("Candidate design", TEAL), ("Prompt interface", PURPLE),
              ("LLM selection", ORANGE), ("Checks + fallback", RED), ("Replanning", GREEN)]
    for i, (t, col) in enumerate(stages):
        x = 1200 + i * 5350
        add_box(doc, p, x, 5300, 4500, 3000, fill=WHITE, line=col, radius=True,
                text=t, size=18, color=NAVY, bold=True)
        if i < len(stages) - 1:
            add_line(doc, p, x + 4550, 6800, x + 5250, 6800, MID, 55)
    add_box(doc, p, 5000, 10100, 23800, 2500, fill=LIGHT, line=TEAL, radius=True,
            text="Observed collaboration = model behaviour × interface × controller × environment",
            size=23, color=NAVY, bold=True)

    p = base_slide(doc, pages, "What this study does not establish", "Limitations",
        "The evaluation is simulation-based and receives symbolic state rather than raw perception. "
        "There is no conventional planning or reinforcement-learning baseline. The number of completed "
        "runs and simulator initialisations limits statistical strength. The natural-language and JSON "
        "conditions differ in both representation and response path. Finally, results belong to the "
        "modified Battery-TA-RWARE system and should not be treated as a direct benchmark comparison "
        "with published TA-RWARE results.")
    limitations = [
        ("SIMULATION", "No physical sensing, hardware uncertainty, or communication delays.", BLUE),
        ("SYMBOLIC STATE", "The LLM does not interpret raw camera or sensor observations.", TEAL),
        ("NO BASELINE", "Feasibility is shown; comparative advantage is not.", RED),
        ("EVIDENCE", "Limited runs and initialisations constrain inference.", ORANGE),
        ("COUPLED PROMPTS", "Input representation and response format are not isolated.", PURPLE),
        ("SYSTEM VERSION", "Results apply to the modified simulator and controller.", GREEN),
    ]
    for i, (head, body, col) in enumerate(limitations):
        x = 1500 + (i % 2) * 16000
        y = 3200 + (i // 2) * 4000
        add_box(doc, p, x, y, 14800, 3200, fill=WHITE, line=col, radius=True)
        add_text(doc, p, x + 500, y + 450, 13800, 500, head, 12, col, True)
        add_text(doc, p, x + 500, y + 1350, 13800, 1150, body, 17, DARK)

    p = base_slide(doc, pages, "Future work", "Outlook",
        "The next step is stronger comparative evidence: more seeds, more runs, broader scenarios, "
        "and explicit non-LLM baselines. The interface can also be studied more carefully by separating "
        "input representation from output constraints. Additional directions include explicit inter-agent "
        "communication, retrieval or tool use when external knowledge is actually needed, and eventual "
        "validation with physical robots and uncertain sensing.")
    future = [
        ("1", "Stronger evaluation", "More runs, seeds, and scenario coverage", BLUE),
        ("2", "Baselines", "Conventional planning and reinforcement learning", RED),
        ("3", "Prompt ablation", "Separate input representation from output format", PURPLE),
        ("4", "Communication", "What, when, and with whom agents should communicate", ORANGE),
        ("5", "Tools and retrieval", "Provide external information only when needed", TEAL),
        ("6", "Physical validation", "Sensor uncertainty, latency, and hardware constraints", GREEN),
    ]
    for i, (n, head, body, col) in enumerate(future):
        x = 1300 + (i % 3) * 10700
        y = 3300 + (i // 3) * 5600
        add_box(doc, p, x, y, 9600, 4600, fill=WHITE, line=col, radius=True)
        add_box(doc, p, x + 300, y + 300, 900, 900, fill=col, line=col, radius=True,
                text=n, size=16, color=WHITE, bold=True)
        add_text(doc, p, x + 1500, y + 500, 7300, 650, head, 19, NAVY, True)
        add_text(doc, p, x + 500, y + 1900, 8500, 1300, body, 16, DARK)

    p = base_slide(doc, pages, "Conclusions", "Conclusion",
        "To conclude, prompted language models supported executable high-level decisions for "
        "heterogeneous collaboration in the tested warehouse simulator. Performance depended strongly "
        "on the selected model, the planning architecture, and the complete prompt interface. Smaller "
        "models remained competitive; shared context generally improved deliveries; and natural language "
        "performed best overall. The contribution is a controlled framework and evidence of feasibility. "
        "Reliability and superiority over established methods remain open questions. Thank you.")
    add_box(doc, p, 2000, 3200, 29800, 2300, fill=NAVY, line=NAVY, radius=True,
            text="Prompted LLMs can support collaborative high-level decisions—\nwithin a carefully designed supervisory system.",
            size=27, color=WHITE, bold=True)
    conclusions = [
        ("MODEL", "Nominal size alone did not predict performance", BLUE),
        ("ARCHITECTURE", "Shared context raised deliveries; centralized used fewer calls", ORANGE),
        ("PROMPT", "Natural language led overall; the effect remained model-dependent", GREEN),
        ("CLAIM", "Feasibility demonstrated; reliability and superiority unresolved", RED),
    ]
    for i, (head, body, col) in enumerate(conclusions):
        x = 2300 + (i % 2) * 15100
        y = 7000 + (i // 2) * 3600
        add_box(doc, p, x, y, 13900, 2800, fill=WHITE, line=col, radius=True)
        add_text(doc, p, x + 500, y + 350, 12900, 450, head, 12, col, True)
        add_text(doc, p, x + 500, y + 1250, 12900, 900, body, 17, NAVY, True)
    add_text(doc, p, 9000, 15100, 16000, 800, "Thank you · Questions", 25, TEAL, True, align=3)

    return pages.getCount()


def main():
    generate_layout_image()
    ctx, smgr, desktop = connect()
    doc = desktop.loadComponentFromURL("private:factory/simpress", "_blank", 0,
                                       (prop("Hidden", True),))
    doc.getDrawPages().getByIndex(0).Width = SW
    doc.getDrawPages().getByIndex(0).Height = SH
    count = build_deck(doc)
    out_url = uno.systemPathToFileUrl(str(OUT))
    doc.storeAsURL(out_url, (prop("FilterName", "Impress MS PowerPoint 2007 XML"),
                             prop("Overwrite", True)))
    doc.close(True)
    print(f"Wrote {OUT} with {count} slides")


if __name__ == "__main__":
    main()
