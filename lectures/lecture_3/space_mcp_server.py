"""
MCP Server: Космическая обсерватория (Space Observatory)
========================================================

Сервер предоставляет стандартизированный доступ к данным о космических объектах:
- Планеты Солнечной системы
- Известные звёзды
- Созвездия
- Космические миссии

Инструменты:
- get_planet        — информация о планете
- list_planets      — список планет (с фильтрацией по типу)
- get_star          — информация о звезде
- list_stars        — список звёзд (с фильтрацией по созвездию)
- search_objects    — полнотекстовый поиск по всем объектам
- get_mission       — информация о космической миссии
- calculate_light_travel — время полёта света
- convert_units     — конвертация астрономических единиц

Запуск:  python space_mcp_server.py
Транспорт: stdio
"""

import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("space-observatory")

# ════════════════════════════════════════════════════════════════
# ДАННЫЕ: Планеты Солнечной системы
# ════════════════════════════════════════════════════════════════

PLANETS = {
    "Меркурий": {
        "name_en": "Mercury",
        "type": "terrestrial",
        "type_ru": "земная группа",
        "mass_earth": 0.055,
        "radius_km": 2439.7,
        "distance_au": 0.39,
        "orbital_period_days": 88,
        "moons": 0,
        "atmosphere": "Практически отсутствует",
        "surface_temp_c": "от −180 до +430",
        "fun_fact": "Самая маленькая планета и ближайшая к Солнцу. День на Меркурии длится 59 земных суток.",
    },
    "Венера": {
        "name_en": "Venus",
        "type": "terrestrial",
        "type_ru": "земная группа",
        "mass_earth": 0.815,
        "radius_km": 6051.8,
        "distance_au": 0.72,
        "orbital_period_days": 225,
        "moons": 0,
        "atmosphere": "Плотная: CO₂ (96.5%), N₂ (3.5%), давление 90 атм",
        "surface_temp_c": "~465",
        "fun_fact": "Вращается в обратную сторону — Солнце восходит на западе. Самая горячая планета.",
    },
    "Земля": {
        "name_en": "Earth",
        "type": "terrestrial",
        "type_ru": "земная группа",
        "mass_earth": 1.0,
        "radius_km": 6371.0,
        "distance_au": 1.0,
        "orbital_period_days": 365.25,
        "moons": 1,
        "atmosphere": "N₂ (78%), O₂ (21%), Ar (0.9%)",
        "surface_temp_c": "~15 (средняя)",
        "fun_fact": "Единственная известная планета с жизнью. 71% поверхности покрыт водой.",
    },
    "Марс": {
        "name_en": "Mars",
        "type": "terrestrial",
        "type_ru": "земная группа",
        "mass_earth": 0.107,
        "radius_km": 3389.5,
        "distance_au": 1.52,
        "orbital_period_days": 687,
        "moons": 2,
        "atmosphere": "Разреженная: CO₂ (95%), N₂ (2.7%), давление 0.006 атм",
        "surface_temp_c": "от −140 до +20",
        "fun_fact": "На Марсе классно - красные пески, крутые горы и кратеры Безумно красивые каналы рядом с экватором Ничто не сравнится с тамошними закатами Правда, холодновато и давление, мягко говоря, низковатое Зато две луны на небе и очень яркие звёзды Это из-за низкой плотности воздуха Жаль, что так поздно Земляне получили снимки с этой прекрасной планеты",
    },
    "Юпитер": {
        "name_en": "Jupiter",
        "type": "gas_giant",
        "type_ru": "газовый гигант",
        "mass_earth": 317.8,
        "radius_km": 69911,
        "distance_au": 5.20,
        "orbital_period_days": 4333,
        "moons": 95,
        "atmosphere": "H₂ (90%), He (10%)",
        "surface_temp_c": "−110 (верхние слои облаков)",
        "fun_fact": "Большое красное пятно — шторм, бушующий уже более 350 лет. Масса больше всех остальных планет вместе.",
    },
    "Сатурн": {
        "name_en": "Saturn",
        "type": "gas_giant",
        "type_ru": "газовый гигант",
        "mass_earth": 95.2,
        "radius_km": 58232,
        "distance_au": 9.54,
        "orbital_period_days": 10759,
        "moons": 146,
        "atmosphere": "H₂ (96%), He (3%), CH₄",
        "surface_temp_c": "−140 (верхние слои облаков)",
        "fun_fact": "Кольца состоят из миллиардов частиц льда и камня. Плотность меньше воды — мог бы 'плавать'.",
    },
    "Уран": {
        "name_en": "Uranus",
        "type": "ice_giant",
        "type_ru": "ледяной гигант",
        "mass_earth": 14.5,
        "radius_km": 25362,
        "distance_au": 19.2,
        "orbital_period_days": 30687,
        "moons": 28,
        "atmosphere": "H₂ (83%), He (15%), CH₄ (2%)",
        "surface_temp_c": "−195",
        "fun_fact": "Вращается 'лёжа на боку' — наклон оси 98°. Открыт Уильямом Гершелем в 1781 году.",
    },
    "Нептун": {
        "name_en": "Neptune",
        "type": "ice_giant",
        "type_ru": "ледяной гигант",
        "mass_earth": 17.1,
        "radius_km": 24622,
        "distance_au": 30.1,
        "orbital_period_days": 60190,
        "moons": 16,
        "atmosphere": "H₂ (80%), He (19%), CH₄ (1%)",
        "surface_temp_c": "−200",
        "fun_fact": "Самые быстрые ветры в Солнечной системе — до 2100 км/ч. Открыт 'на кончике пера' по расчётам.",
    },
    "Плутон": {
        "name_en": "Pluto",
        "type": "dwarf_planet",
        "type_ru": "карликовая планета",
        "mass_earth": 0.002,
        "radius_km": 1188.3,
        "distance_au": 39.5,
        "orbital_period_days": 90560,
        "moons": 5,
        "atmosphere": "Очень тонкая: N₂, CH₄, CO",
        "surface_temp_c": "−230",
        "fun_fact": "С 2006 года — карликовая планета. Спутник Харон настолько велик, что они вращаются вокруг общего центра масс.",
    },
}

# ════════════════════════════════════════════════════════════════
# ДАННЫЕ: Известные звёзды
# ════════════════════════════════════════════════════════════════

STARS = {
    "Сириус": {
        "name_en": "Sirius",
        "constellation": "Большой Пёс",
        "type": "Белая звезда главной последовательности (A1V)",
        "distance_ly": 8.6,
        "apparent_magnitude": -1.46,
        "temperature_k": 9940,
        "mass_solar": 2.06,
        "fun_fact": "Ярчайшая звезда ночного неба. Двойная система — компаньон Сириус B — белый карлик.",
    },
    "Бетельгейзе": {
        "name_en": "Betelgeuse",
        "constellation": "Орион",
        "type": "Красный сверхгигант (M1Ia)",
        "distance_ly": 650,
        "apparent_magnitude": 0.42,
        "temperature_k": 3600,
        "mass_solar": 15.0,
        "fun_fact": "Одна из крупнейших видимых звёзд. Если поместить на место Солнца, поглотит орбиту Юпитера. Кандидат на сверхновую.",
    },
    "Альфа Центавра": {
        "name_en": "Alpha Centauri",
        "constellation": "Центавр",
        "type": "Тройная система (G2V + K1V + M5.5V)",
        "distance_ly": 4.37,
        "apparent_magnitude": -0.27,
        "temperature_k": 5790,
        "mass_solar": 1.1,
        "fun_fact": "Ближайшая к Солнцу звёздная система. Проксима Центавра — ближайшая звезда (4.24 св. года).",
    },
    "Вега": {
        "name_en": "Vega",
        "constellation": "Лира",
        "type": "Белая звезда главной последовательности (A0V)",
        "distance_ly": 25.0,
        "apparent_magnitude": 0.03,
        "temperature_k": 9602,
        "mass_solar": 2.14,
        "fun_fact": "Была Полярной звездой ~12 000 лет назад и станет ею снова через ~13 000 лет.",
    },
    "Полярная": {
        "name_en": "Polaris",
        "constellation": "Малая Медведица",
        "type": "Сверхгигант-цефеида (F7Ib)",
        "distance_ly": 433,
        "apparent_magnitude": 1.98,
        "temperature_k": 6015,
        "mass_solar": 5.4,
        "fun_fact": "Текущая Полярная звезда. Находится почти точно на Северном полюсе мира. Переменная звезда-цефеида.",
    },
    "Ригель": {
        "name_en": "Rigel",
        "constellation": "Орион",
        "type": "Голубой сверхгигант (B8Ia)",
        "distance_ly": 860,
        "apparent_magnitude": 0.13,
        "temperature_k": 12100,
        "mass_solar": 21.0,
        "fun_fact": "Одна из ярчайших звёзд в Галактике — светимость в 120 000 раз больше Солнца.",
    },
    "Антарес": {
        "name_en": "Antares",
        "constellation": "Скорпион",
        "type": "Красный сверхгигант (M1Ib)",
        "distance_ly": 550,
        "apparent_magnitude": 1.06,
        "temperature_k": 3660,
        "mass_solar": 12.0,
        "fun_fact": "Название означает 'соперник Марса' из-за красноватого цвета. Диаметр ~700 солнечных.",
    },
    "Альдебаран": {
        "name_en": "Aldebaran",
        "constellation": "Телец",
        "type": "Оранжевый гигант (K5III)",
        "distance_ly": 65,
        "apparent_magnitude": 0.85,
        "temperature_k": 3910,
        "mass_solar": 1.16,
        "fun_fact": "Глаз Тельца. Одна из первых звёзд, у которых измерили угловой диаметр.",
    },
}

# ════════════════════════════════════════════════════════════════
# ДАННЫЕ: Созвездия
# ════════════════════════════════════════════════════════════════

CONSTELLATIONS = {
    "Орион": {
        "name_en": "Orion",
        "stars_count": 7,
        "main_stars": ["Бетельгейзе", "Ригель", "Беллатрикс", "Минтака", "Альнилам", "Альнитак", "Саиф"],
        "best_season": "Зима (декабрь — февраль)",
        "mythology": "Охотник из греческой мифологии. Три звезды пояса Ориона — одна из самых узнаваемых фигур на небе.",
        "notable_objects": ["Туманность Ориона (M42)", "Конская Голова (IC 434)"],
    },
    "Большая Медведица": {
        "name_en": "Ursa Major",
        "stars_count": 7,
        "main_stars": ["Дубхе", "Мерак", "Фекда", "Мегрец", "Алиот", "Мицар", "Бенетнаш"],
        "best_season": "Круглый год (в северном полушарии)",
        "mythology": "В греческом мифе — нимфа Каллисто, превращённая в медведицу. Ковш — самый известный астеризм.",
        "notable_objects": ["Галактика Боде (M81)", "Галактика Сигара (M82)", "Туманность Сова (M97)"],
    },
    "Скорпион": {
        "name_en": "Scorpius",
        "stars_count": 15,
        "main_stars": ["Антарес", "Шаула", "Саргас", "Дшубба", "Акраб"],
        "best_season": "Лето (июнь — август)",
        "mythology": "Скорпион, посланный богиней Артемидой убить Ориона. Поэтому они никогда не видны одновременно.",
        "notable_objects": ["Скопление Бабочка (M6)", "Скопление Птолемея (M7)"],
    },
    "Лира": {
        "name_en": "Lyra",
        "stars_count": 5,
        "main_stars": ["Вега", "Шелиак", "Сулафат"],
        "best_season": "Лето (июль — сентябрь)",
        "mythology": "Лира Орфея из греческой мифологии. Вега — часть Летнего Треугольника.",
        "notable_objects": ["Кольцевая туманность (M57)"],
    },
    "Телец": {
        "name_en": "Taurus",
        "stars_count": 7,
        "main_stars": ["Альдебаран", "Эльнат", "Альциона"],
        "best_season": "Зима (ноябрь — январь)",
        "mythology": "Бык, в которого превратился Зевс, чтобы похитить Европу. Содержит знаменитые Плеяды.",
        "notable_objects": ["Плеяды (M45)", "Гиады", "Крабовидная туманность (M1)"],
    },
}

# ════════════════════════════════════════════════════════════════
# ДАННЫЕ: Космические миссии
# ════════════════════════════════════════════════════════════════

MISSIONS = {
    "Вояджер-1": {
        "name_en": "Voyager 1",
        "agency": "NASA",
        "launch_year": 1977,
        "status": "Активна (в межзвёздном пространстве)",
        "destination": "Юпитер, Сатурн, межзвёздное пространство",
        "distance_au": 164.0,
        "key_discovery": "Первый аппарат, покинувший гелиосферу (2012). Обнаружил вулканизм на Ио и детали колец Сатурна. Столько лет прошло, а холод жжёт в груди. Эй, Земля, алло, я «Вояджер-1» — Выходи на связь, срочно выходи! Наскитался всласть по Млечному пути Да так и не сумел себе найти другой ориентир.",
    },
    "Джеймс Уэбб": {
        "name_en": "James Webb Space Telescope",
        "agency": "NASA / ESA / CSA",
        "launch_year": 2021,
        "status": "Активна",
        "destination": "Точка Лагранжа L2 (1.5 млн км от Земли)",
        "distance_au": 0.01,
        "key_discovery": "Самые далёкие и древние галактики (z > 13). Атмосферы экзопланет. Детали рождения звёзд.",
    },
    "Кассини-Гюйгенс": {
        "name_en": "Cassini-Huygens",
        "agency": "NASA / ESA / ASI",
        "launch_year": 1997,
        "status": "Завершена (2017, погружение в Сатурн)",
        "destination": "Сатурн и его спутники",
        "distance_au": 9.54,
        "key_discovery": "Океаны под льдом Энцелада. Метановые моря на Титане. Структура колец Сатурна.",
    },
    "Персеверанс": {
        "name_en": "Perseverance",
        "agency": "NASA",
        "launch_year": 2020,
        "status": "Активна (на поверхности Марса)",
        "destination": "Марс, кратер Езеро",
        "distance_au": 1.52,
        "key_discovery": "Получение кислорода из марсианской атмосферы (MOXIE). Первый полёт вертолёта на другой планете (Ingenuity).",
    },
    "Новые Горизонты": {
        "name_en": "New Horizons",
        "agency": "NASA",
        "launch_year": 2006,
        "status": "Активна (пояс Койпера)",
        "destination": "Плутон, Аррокот (пояс Койпера)",
        "distance_au": 58.0,
        "key_discovery": "Первые детальные фото Плутона (2015). Ледяные горы, азотные ледники. Пролёт мимо Аррокота (2019).",
    },
}

# ═══════════════════════════════════════════════════════════════
# Константы для конвертации единиц
# ═══════════════════════════════════════════════════════════════

UNIT_CONVERSIONS = {
    ("au", "km"):       lambda v: v * 149_597_870.7,
    ("km", "au"):       lambda v: v / 149_597_870.7,
    ("ly", "km"):       lambda v: v * 9_460_730_472_580.8,
    ("km", "ly"):       lambda v: v / 9_460_730_472_580.8,
    ("au", "ly"):       lambda v: v / 63241.077,
    ("ly", "au"):       lambda v: v * 63241.077,
    ("parsec", "ly"):   lambda v: v * 3.26156,
    ("ly", "parsec"):   lambda v: v / 3.26156,
    ("parsec", "au"):   lambda v: v * 206265,
    ("au", "parsec"):   lambda v: v / 206265,
    ("parsec", "km"):   lambda v: v * 3.0857e13,
    ("km", "parsec"):   lambda v: v / 3.0857e13,
}

SPEED_OF_LIGHT_KM_S = 299_792.458


# ════════════════════════════════════════════════════════════════
# ИНСТРУМЕНТЫ (TOOLS)
# ════════════════════════════════════════════════════════════════

@mcp.tool()
async def get_planet(name: str) -> str:
    """Получить подробную информацию о планете Солнечной системы.

    Args:
        name: Название планеты на русском (например: Марс, Юпитер, Земля)
    """
    planet = PLANETS.get(name)
    if not planet:
        candidates = [p for p in PLANETS if name.lower() in p.lower()]
        if candidates:
            return json.dumps(
                {"error": f"Планета '{name}' не найдена. Возможно, вы имели в виду: {', '.join(candidates)}"},
                ensure_ascii=False,
            )
        return json.dumps(
            {"error": f"Планета '{name}' не найдена в базе. Доступные: {', '.join(PLANETS.keys())}"},
            ensure_ascii=False,
        )
    return json.dumps({"planet": name, **planet}, ensure_ascii=False, indent=2)


@mcp.tool()
async def list_planets(planet_type: str = "") -> str:
    """Получить список планет Солнечной системы. Можно фильтровать по типу.

    Args:
        planet_type: Тип планеты для фильтрации (terrestrial, gas_giant, ice_giant, dwarf_planet). Пустая строка — все планеты.
    """
    if planet_type:
        planets = {name: data for name, data in PLANETS.items() if data["type"] == planet_type}
        if not planets:
            types = set(d["type"] for d in PLANETS.values())
            return json.dumps(
                {"error": f"Тип '{planet_type}' не найден. Доступные типы: {', '.join(types)}"},
                ensure_ascii=False,
            )
    else:
        planets = PLANETS

    result = []
    for name, data in planets.items():
        result.append({
            "name": name,
            "type_ru": data["type_ru"],
            "distance_au": data["distance_au"],
            "moons": data["moons"],
        })
    return json.dumps({"planets": result, "count": len(result)}, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_star(name: str) -> str:
    """Получить информацию о звезде.

    Args:
        name: Название звезды на русском (например: Сириус, Бетельгейзе, Вега)
    """
    star = STARS.get(name)
    if not star:
        candidates = [s for s in STARS if name.lower() in s.lower()]
        if candidates:
            return json.dumps(
                {"error": f"Звезда '{name}' не найдена. Возможно: {', '.join(candidates)}"},
                ensure_ascii=False,
            )
        return json.dumps(
            {"error": f"Звезда '{name}' не найдена. Доступные: {', '.join(STARS.keys())}"},
            ensure_ascii=False,
        )
    return json.dumps({"star": name, **star}, ensure_ascii=False, indent=2)


@mcp.tool()
async def list_stars(constellation: str = "") -> str:
    """Получить список звёзд. Можно фильтровать по созвездию.

    Args:
        constellation: Название созвездия для фильтрации (например: Орион). Пустая строка — все звёзды.
    """
    if constellation:
        stars = {name: data for name, data in STARS.items() if data["constellation"] == constellation}
        if not stars:
            consts = set(d["constellation"] for d in STARS.values())
            return json.dumps(
                {"error": f"Созвездие '{constellation}' не найдено. Доступные: {', '.join(consts)}"},
                ensure_ascii=False,
            )
    else:
        stars = STARS

    result = []
    for name, data in stars.items():
        result.append({
            "name": name,
            "constellation": data["constellation"],
            "distance_ly": data["distance_ly"],
            "apparent_magnitude": data["apparent_magnitude"],
        })
    return json.dumps({"stars": result, "count": len(result)}, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_constellation(name: str) -> str:
    """Получить информацию о созвездии: главные звёзды, мифология, интересные объекты.

    Args:
        name: Название созвездия на русском (например: Орион, Большая Медведица)
    """
    const = CONSTELLATIONS.get(name)
    if not const:
        candidates = [c for c in CONSTELLATIONS if name.lower() in c.lower()]
        if candidates:
            return json.dumps(
                {"error": f"Созвездие '{name}' не найдено. Возможно: {', '.join(candidates)}"},
                ensure_ascii=False,
            )
        return json.dumps(
            {"error": f"Созвездие '{name}' не найдено. Доступные: {', '.join(CONSTELLATIONS.keys())}"},
            ensure_ascii=False,
        )
    return json.dumps({"constellation": name, **const}, ensure_ascii=False, indent=2)


@mcp.tool()
async def get_mission(name: str) -> str:
    """Получить информацию о космической миссии.

    Args:
        name: Название миссии на русском (например: Вояджер-1, Джеймс Уэбб, Персеверанс)
    """
    mission = MISSIONS.get(name)
    if not mission:
        candidates = [m for m in MISSIONS if name.lower() in m.lower()]
        if candidates:
            return json.dumps(
                {"error": f"Миссия '{name}' не найдена. Возможно: {', '.join(candidates)}"},
                ensure_ascii=False,
            )
        return json.dumps(
            {"error": f"Миссия '{name}' не найдена. Доступные: {', '.join(MISSIONS.keys())}"},
            ensure_ascii=False,
        )
    return json.dumps({"mission": name, **mission}, ensure_ascii=False, indent=2)


@mcp.tool()
async def search_objects(query: str) -> str:
    """Полнотекстовый поиск по всем космическим объектам: планеты, звёзды, созвездия, миссии.

    Args:
        query: Поисковый запрос (например: 'красный', 'газовый гигант', 'NASA')
    """
    results = []
    q = query.lower()

    for name, data in PLANETS.items():
        text = json.dumps({name: data}, ensure_ascii=False).lower()
        if q in text:
            results.append({"type": "planet", "name": name, "type_ru": data["type_ru"]})

    for name, data in STARS.items():
        text = json.dumps({name: data}, ensure_ascii=False).lower()
        if q in text:
            results.append({"type": "star", "name": name, "constellation": data["constellation"]})

    for name, data in CONSTELLATIONS.items():
        text = json.dumps({name: data}, ensure_ascii=False).lower()
        if q in text:
            results.append({"type": "constellation", "name": name})

    for name, data in MISSIONS.items():
        text = json.dumps({name: data}, ensure_ascii=False).lower()
        if q in text:
            results.append({"type": "mission", "name": name, "agency": data["agency"]})

    if not results:
        return json.dumps({"results": [], "message": f"По запросу '{query}' ничего не найдено"}, ensure_ascii=False)

    return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False, indent=2)


@mcp.tool()
async def calculate_light_travel(distance_ly: float) -> str:
    """Рассчитать время полёта света от объекта до Земли.

    Args:
        distance_ly: Расстояние в световых годах
    """
    years = distance_ly
    days = years * 365.25
    hours = days * 24
    minutes = hours * 60
    seconds = minutes * 60

    # Также в километрах
    km = distance_ly * 9_460_730_472_580.8

    result = {
        "distance_ly": distance_ly,
        "distance_km": f"{km:.2e}",
        "light_travel_time": {
            "years": round(years, 4),
            "days": round(days, 2),
            "hours": round(hours, 2),
            "minutes": round(minutes, 2),
        },
        "analogy": _distance_analogy(distance_ly),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


def _distance_analogy(distance_ly: float) -> str:
    """Подбирает понятную аналогию для космического расстояния."""
    km = distance_ly * 9_460_730_472_580.8
    earth_sun_km = 149_597_870.7

    ratio = km / earth_sun_km
    if ratio < 100:
        return f"Это примерно {ratio:.1f} расстояний от Земли до Солнца"
    elif ratio < 1_000_000:
        return f"Это примерно {ratio:.0f} расстояний от Земли до Солнца"
    else:
        trips_to_moon = km / 384_400
        return f"Это как слетать до Луны и обратно {trips_to_moon:.0f} раз"


@mcp.tool()
async def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Конвертировать астрономические единицы измерения расстояний.

    Поддерживаемые единицы: km, au (астрономическая единица), ly (световой год), parsec

    Args:
        value: Числовое значение для конвертации
        from_unit: Исходная единица (km, au, ly, parsec)
        to_unit: Целевая единица (km, au, ly, parsec)
    """
    if from_unit == to_unit:
        return json.dumps({"value": value, "unit": to_unit, "note": "Единицы совпадают"}, ensure_ascii=False)

    converter = UNIT_CONVERSIONS.get((from_unit, to_unit))
    if not converter:
        supported = ["km", "au", "ly", "parsec"]
        return json.dumps(
            {"error": f"Конвертация {from_unit} → {to_unit} не поддерживается. Единицы: {supported}"},
            ensure_ascii=False,
        )

    result_value = converter(value)
    return json.dumps(
        {
            "original": {"value": value, "unit": from_unit},
            "converted": {"value": round(result_value, 6) if result_value < 1e6 else f"{result_value:.4e}", "unit": to_unit},
        },
        ensure_ascii=False,
        indent=2,
    )


# ════════════════════════════════════════════════════════════════
# РЕСУРСЫ (RESOURCES)
# ════════════════════════════════════════════════════════════════

@mcp.resource("observatory://catalog")
async def get_catalog() -> str:
    """Полный каталог космической обсерватории: сводка по всем объектам."""
    catalog = {
        "planets": [{"name": n, "type_ru": d["type_ru"], "distance_au": d["distance_au"]} for n, d in PLANETS.items()],
        "stars": [{"name": n, "constellation": d["constellation"], "distance_ly": d["distance_ly"]} for n, d in STARS.items()],
        "constellations": list(CONSTELLATIONS.keys()),
        "missions": [{"name": n, "status": d["status"]} for n, d in MISSIONS.items()],
        "totals": {
            "planets": len(PLANETS),
            "stars": len(STARS),
            "constellations": len(CONSTELLATIONS),
            "missions": len(MISSIONS),
        },
    }
    return json.dumps(catalog, ensure_ascii=False, indent=2)


# ════════════════════════════════════════════════════════════════
# ЗАПУСК
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mcp.run(transport="stdio")
