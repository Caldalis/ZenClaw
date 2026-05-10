#!/usr/bin/env python3
"""
Weather Scraper - 爬取北京天气数据并保存到 JSON 文件。
使用 urllib.request 获取 wttr.in 的 JSON 格式天气数据。
"""

import json
import urllib.request
import urllib.error


URL = "https://wttr.in/Beijing?format=j1"
OUTPUT_FILE = "weather_data.json"


def fetch_weather(url: str) -> dict:
    """从指定 URL 获取天气 JSON 数据。"""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "WeatherScraper/1.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        if response.status != 200:
            raise ValueError(f"HTTP 请求失败，状态码: {response.status}")
        data = response.read().decode("utf-8")
    return json.loads(data)


def extract_weather_info(raw_data: dict) -> dict:
    """从原始 JSON 数据中提取所需的天气字段。"""
    # 获取当前天气部分
    current = raw_data.get("current_condition", [{}])[0]

    # 温度 (°C)
    temperature = current.get("temp_C", "N/A")

    # 湿度 (%)
    humidity = current.get("humidity", "N/A")

    # 风速 (km/h)
    wind_speed = current.get("windspeedKmph", "N/A")

    # 天气状况描述
    weather_condition = current.get("weatherDesc", [{}])[0].get("value", "N/A")

    # AQI - 从 atmosphere 节点获取
    aqi_value = "N/A"
    try:
        atmosphere = raw_data.get("atmosphere", {})
        # wttr.in 在某些地区会提供 aqi 字段
        if "aqi" in atmosphere:
            aqi_value = atmosphere["aqi"]
        else:
            # 尝试从 weatherParam 中查找 AQI 相关数据
            for param in raw_data.get("weather", [{}])[0].get("hourly", []):
                aqi_in_hourly = param.get("aqi", None)
                if aqi_in_hourly is not None:
                    aqi_value = aqi_in_hourly
                    break
    except (IndexError, TypeError, KeyError):
        aqi_value = "N/A"

    return {
        "temperature": f"{temperature} °C",
        "humidity": f"{humidity} %",
        "wind_speed": f"{wind_speed} km/h",
        "AQI": aqi_value,
        "weather_condition": weather_condition,
        "timestamp": raw_data.get("nearest_area", [{}])[0].get("areaName", [{}])[0].get("value", "Unknown") + ", " + raw_data.get("nearest_area", [{}])[0].get("country", [{}])[0].get("value", ""),
    }


def save_to_json(data: dict, filepath: str) -> None:
    """将数据保存为 JSON 文件。"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"天气数据已保存到 {filepath}")


def main():
    """主函数：爬取天气数据并保存。"""
    print(f"正在爬取 {URL} ...")
    try:
        # 1. 获取原始数据
        raw_data = fetch_weather(URL)

        # 2. 提取所需信息
        weather_info = extract_weather_info(raw_data)

        # 3. 保存到 JSON
        save_to_json(weather_info, OUTPUT_FILE)

        # 4. 打印结果
        print("\n--- 北京天气信息 ---")
        for key, value in weather_info.items():
            print(f"  {key}: {value}")

    except urllib.error.URLError as e:
        print(f"网络请求失败: {e.reason}")
    except urllib.error.HTTPError as e:
        print(f"HTTP 错误: {e.code} - {e.reason}")
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == "__main__":
    main()
