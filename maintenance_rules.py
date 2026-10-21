"""Transparent, rule-based risk categories and maintenance suggestions.

Thresholds follow the failure-mode descriptions in the AI4I 2020 dataset documentation
(tool wear, heat dissipation, power, and overstrain failures).
"""
import math

OVERSTRAIN_LIMITS = {"L": 11000, "M": 12000, "H": 13000}   # tool wear [min] x torque [Nm]


def risk_category(prob, threshold):
    if prob >= threshold:
        return "High"
    if prob >= threshold / 2:
        return "Medium"
    return "Low"


def maintenance_suggestions(rec, prob, threshold):
    """rec: dict with air_temp, process_temp, rpm, torque, tool_wear, type."""
    tips = []
    power = rec["torque"] * rec["rpm"] * 2 * math.pi / 60       # watts
    delta_t = rec["process_temp"] - rec["air_temp"]
    strain = rec["tool_wear"] * rec["torque"]
    limit = OVERSTRAIN_LIMITS.get(rec["type"], 12000)

    if rec["tool_wear"] >= 200:
        tips.append("🔧 Tool wear is very high (>= 200 min): inspect and schedule tool replacement.")
    elif rec["tool_wear"] >= 160:
        tips.append("🔧 Tool wear is approaching the critical range: plan replacement at next stop.")
    if delta_t < 8.6 and rec["rpm"] < 1380:
        tips.append("🌡️ Small temperature difference at low speed: check cooling / heat-dissipation system.")
    if rec["process_temp"] >= 313.5:
        tips.append("🌡️ Process temperature is high: verify coolant flow and ventilation.")
    if power < 3500:
        tips.append(f"⚡ Power is unusually low ({power:,.0f} W): check drive, load and speed settings.")
    elif power > 9000:
        tips.append(f"⚡ Power is unusually high ({power:,.0f} W): reduce load or inspect motor/drive.")
    if strain > limit * 0.9:
        tips.append(f"🏋️ Tool wear x torque = {strain:,.0f} min·Nm (limit for type {rec['type']} ≈ {limit:,}): overstrain risk — lower torque or replace tool.")
    if rec["torque"] > 60:
        tips.append("🏋️ Torque is very high: review cutting load and fixturing.")

    level = risk_category(prob, threshold)
    if level == "High" and not tips:
        tips.append("⚠️ Model flags high risk without a single dominant cause: schedule a general inspection.")
    if level == "Medium" and not tips:
        tips.append("👀 Elevated risk: monitor this machine more closely.")
    if level == "Low" and not tips:
        tips.append("✅ No action needed: operating conditions look normal.")
    return tips
