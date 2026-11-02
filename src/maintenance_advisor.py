
# Reference dataset screening thresholds

THRESHOLDS = {
    "air_temp_high": 303.5,
    "process_temp_high": 312.5,
    "speed_low": 1332.0,
    "speed_high": 1868.05,
    "torque_high": 56.1,
    "tool_wear_high": 195.0
}


def recommend_maintenance(machine_data):

    recommendations = []
    flags = []

    # 1. Tool Wear Analysis
    if machine_data["tool_wear"] >= THRESHOLDS["tool_wear_high"]:

        flags.append("High tool wear relative to reference dataset")

        recommendations.append(
            "Inspect the cutting tool and review its replacement schedule."
        )

    # 2. Temperature Analysis
    if (
        machine_data["air_temperature"] >= THRESHOLDS["air_temp_high"]
        or
        machine_data["process_temperature"] >= THRESHOLDS["process_temp_high"]
    ):

        flags.append("Temperature is in the upper reference-dataset range")

        recommendations.append(
            "Check cooling airflow, coolant circulation, and temperature sensors."
        )

    # 3. Rotational Speed Analysis
    speed = machine_data["rotational_speed"]

    if (
        speed <= THRESHOLDS["speed_low"]
        or
        speed >= THRESHOLDS["speed_high"]
    ):

        flags.append(
            "Rotational speed is outside the central 90% reference range"
        )

        recommendations.append(
            "Inspect the motor, drive settings, and speed-control system."
        )

    # 4. Torque Analysis
    if machine_data["torque"] >= THRESHOLDS["torque_high"]:

        flags.append("Torque is in the upper reference-dataset range")

        recommendations.append(
            "Inspect mechanical load, bearings, and drivetrain for abnormal resistance."
        )

    # 5. Final Status
    if not recommendations:

        recommendations.append(
            "No configured screening rule was triggered. Continue routine monitoring."
        )

    status = "Warning" if flags else "Normal screening"

    return {
        "status": status,
        "flags": flags,
        "recommendations": recommendations,
        "thresholds_used": THRESHOLDS,
        "disclaimer": (
            "Rule-based screening only; this is not a confirmed "
            "failure diagnosis."
        )
    }
