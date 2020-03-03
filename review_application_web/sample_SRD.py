import requests

url = 'http://192.9.66.224:8000/application/srd/'

param_1 = {'review': '5 days it takes this lethargic water company to turn our water on in our new home in Surprise!'
                      '  If you do have to deal with their customer service reps, you will see the perfect example of '
                      'a company with no standards.'}
param_2 = {'review': 'Just kidding. I just took my car to a real mechanic, and he asked if I actually paid money '
                      'for the shitty rig job Mike did on my car, and was surprised my car was actually still running. '
                      'Thanks for taking advantage of a girl in distress, jerk.'}
param_3 = {'review': 'it is nice restaurant'}

params = [param_1, param_2, param_3]

for param in params:
    response = requests.get(url, params=param)
    print(response.text, '\n\n')
