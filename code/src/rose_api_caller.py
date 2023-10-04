import requests
import json


# TODO: FIX THIS REST SERVICE
def fetch_tree():

	API_URL = "https://bibiserv.cebitec.uni-bielefeld.de/applications/rose/pages/rose_fct_AA_result.jsf"
	API_STATUS_URL = "http://bibiserv2.cebitec.uni-bielefeld.de/rose/rose_fct_AA/statuscode"
	API_RESPONSE_URL = "http://bibiserv2.cebitec.uni-bielefeld.de/rose/rose_fct_AA/response"

	headers = {
	    "Content-Type": "application/json"
	}

	f = open("data/api/rose_fct_AA.example.json", "r")
	data = json.load(f)
	response = requests.post(API_URL, json=data, headers=headers)

	ID = response.text
	print(f"Job ID: {ID}")

	headers = {
	    "Content-Type": "text/plain"
	}

	status = None
	while status != 600:
		response = requests.post(API_STATUS_URL, data=ID, headers=headers)
		status = response.status_code
		print(f"Waiting for process to finish. Got status code: {status}")
		time.sleep(0.5)

	response = requests.post(API_RESPONSE_URL, data=ID, headers=headers)

	f.close()

	return response.data


if __name__ == "__main__":
	print(fetch_tree())