import requests
import json

def get_locality_weather(locality_id):
    """
    Fetch weather data for a given locality using its locality_id.

    Parameters:
    locality_id (int): The ID of the locality for which to retrieve weather data.

    Returns:
    str: The formatted weather data as a JSON string if successful, or an error message.
    """
    url = "https://weatherunion.com/gw/weather/external/v0/get_locality_weather_data"
    params = {'locality_id': locality_id}
    headers = {"x-zomato-api-key": YOUR_API_KEY}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        try:
            parsed = json.loads(response.text)
            return json.dumps(parsed, indent=4)
        except json.JSONDecodeError:
            return response.text
    else:
        return f"Failed to retrieve data: {response.status_code}\n{response.text}"
    

if __name__ =='__main__':
    print(get_locality_weather("ZWL001156"))
