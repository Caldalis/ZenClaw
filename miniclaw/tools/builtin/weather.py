"""
天气查询技能 — 内置技能示例
使用 Open-Meteo 免费 API 查询天气（无需 API Key）
"""

from __future__ import annotations

from typing import Any

import httpx

from miniclaw.tools.base import Tool


# Open-Meteo 地理编码 API
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
# Open-Meteo 天气 API
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


class WeatherSkill(Tool):
    """天气查询 — 查询指定城市的当前天气"""

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "查询指定城市的当前天气信息，包括温度、湿度、风速等。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如 '北京'、'Tokyo'、'New York'",
                },
            },
            "required": ["city"],
        }

    async def execute(self, **kwargs: Any) -> str:
        city = kwargs.get("city", "")
        if not city:
            return "请提供城市名称"

        async with httpx.AsyncClient(timeout=10) as client:
            # Step 1: 地理编码 — 城市名 → 经纬度
            geo_resp = await client.get(GEOCODE_URL, params={"name": city, "count": 1, "language": "zh"})
            geo_data = geo_resp.json()

            if not geo_data.get("results"):
                return f"未找到城市: {city}"

            location = geo_data["results"][0]
            lat, lon = location["latitude"], location["longitude"]
            city_name = location.get("name", city)
            country = location.get("country", "")

            # Step 2: 查询天气
            weather_resp = await client.get(WEATHER_URL, params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "timezone": "auto",
            })
            weather_data = weather_resp.json()
            current = weather_data.get("current", {})

            temp = current.get("temperature_2m", "N/A")
            humidity = current.get("relative_humidity_2m", "N/A")
            wind = current.get("wind_speed_10m", "N/A")
            code = current.get("weather_code", -1)
            condition = _weather_code_to_text(code)

            return (
                f"{city_name}, {country}\n"
                f"温度: {temp}°C\n"
                f"湿度: {humidity}%\n"
                f"风速: {wind} km/h\n"
                f"天气: {condition}"
            )


def _weather_code_to_text(code: int) -> str:
    """WMO 天气代码转文字描述"""
    mapping = {
        0: "晴天", 1: "大部分晴", 2: "局部多云", 3: "多云",
        45: "雾", 48: "霜雾",
        51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
        61: "小雨", 63: "中雨", 65: "大雨",
        71: "小雪", 73: "中雪", 75: "大雪",
        80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
        95: "雷暴", 96: "雷暴+小冰雹", 99: "雷暴+大冰雹",
    }
    return mapping.get(code, f"未知({code})")
