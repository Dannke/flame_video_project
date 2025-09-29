MASK_PARAMS = {
    "buffer_size": 15, # Размер буфера кадров
    "brightness_thresh": 215, # Порог яркости (210-240)
    "saturation_thresh": 115, # Порог насыщенности (100-140)
    "flicker_weight": 0.25, # Вес мерцания (0-0.5)
    "min_flicker_frames": 5, # Минимум кадров для анализа мерцания
    "use_color_filter": True, # Использовать цветовой фильтр
    "confidence_threshold": 15.0, # Минимальная уверенность для "пламени"
    "flame_percent_threshold": 0.15, # Минимальный процент для "пламени"
    "negative_keep_ratio": 0.05, # Доля % кадров без пламени
}
