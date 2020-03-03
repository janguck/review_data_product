import requests

url = 'http://192.9.66.224:8000/application/fead/'

param_1 = {'review': 'So disappointing. Was craving bubble tea and this place seemed legit. '
                     'Ordered an almond milk tea and when I took a drink, my mouth was flooded with imitation almond '
                     'flavoring. I got an instant headache and ended up throwing it alway.',
           'description': 'Coffee & Tea, Food, Bubble Tea, Juice Bars & Smoothies'}
param_2 = {'review': '''I've cracked my phone a few times and I've been to a few different iPhone repair spots, 
this one was the quickest! It was ready in 20 minutes and my phone looks brand new! 
They even hooked me up with a screen protector''',
           'description': 'Data Recovery, Computers, IT Services & Computer Repair, Mobile Phone Accessories, '
                          'Local Services, Shopping, Mobile Phone Repair, Electronics Repair'}

params = [param_1, param_2]

for param in params:
    response = requests.get(url, params=param)
    print(response.text, '\n\n')
