OOD_SCENARIOS = {
    'nominal': {
        'mass_multiplier': 1.0,
        'motor_efficiency': [1.0, 1.0, 1.0, 1.0],
        'wind_speed': 0.0,
        'description': 'Ideal conditions - no disturbances'
    },
    'heavy_payload': {
        'mass_multiplier': 1.2,
        'motor_efficiency': [1.0, 1.0, 1.0, 1.0],
        'wind_speed': 0.0,
        'description': 'Carrying package (+20% mass)'
    },
    'light_payload': {
        'mass_multiplier': 0.8,
        'motor_efficiency': [1.0, 1.0, 1.0, 1.0],
        'wind_speed': 0.0,
        'description': 'After delivery (-20% mass)'
    },
    'damaged_motor': {
        'mass_multiplier': 1.0,
        'motor_efficiency': [1.0, 1.0, 0.7, 1.0],
        'wind_speed': 0.0,
        'description': 'Motor 3 degraded (70% efficiency)'
    },
    'strong_wind': {
        'mass_multiplier': 1.0,
        'motor_efficiency': [1.0, 1.0, 1.0, 1.0],
        'wind_speed': 2.0,
        'description': 'Wind gusts (2 m/s)'
    },
    'combined_worst': {
        'mass_multiplier': 1.2,
        'motor_efficiency': [1.0, 1.0, 0.7, 1.0],
        'wind_speed': 1.5,
        'description': 'Heavy payload + damaged motor + wind'
    },
    'critical_motor_failure': {
        'mass_multiplier': 1.0,
        'motor_efficiency': [1.0, 1.0, 0.5, 1.0],
        'wind_speed': 0.0,
        'description': 'Emergency: one motor at 50% efficiency (battery critical/motor failure)'
    }
}


def get_scenario(name):
    if name not in OOD_SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(OOD_SCENARIOS.keys())}")
    return OOD_SCENARIOS[name]
