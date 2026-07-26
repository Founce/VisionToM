import json_repair

def rm(choices, answer, response):
    answer = answer.lower()
    response = response.lower()

    try:
        response_json = json_repair.loads(response)
        if isinstance(response_json, list):
            if len(response_json) > 1:
                raise ValueError(choices, "Multiple responses found.")
            else:
                response_json = response_json[0]
        if isinstance(response_json, int) or isinstance(response_json, float):
            response_json = str(response_json)
    except:
        return 'Error'

    if 'choice' in response_json:
        response_choice = response_json['choice'].lower()
        if response_choice == answer:
            return 'Right'
        else:
            return 'Wrong'
    else:
        return 'Error'
