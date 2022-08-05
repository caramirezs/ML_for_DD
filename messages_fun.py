import requests

ACCESS_TOKEN = 'EAAHZCimB106QBAHZAeJ3yZCdY9T984ERNSq1AGEQS6mmXn0dCSgLwvFg0dG3nMtMn2t1NkBGmPVhYjeCQ0iVOReuko35U1q3qydtJGZBB1u0ZAhxxE9BtLalAVDtiIqnWtZBdDjC5P5aGkrSXotRwGWAeQ6ZBeAZB4idNHMH0DqGKVBf6w5H1aPZCglLwqnK22LYZD'
FROM_PHONE_NUMBER_ID = '102045155949896'
PHONE_NUMBER = '573164185522'
version = 'v14.0'
endpoint = f'https://graph.facebook.com/{version}/{FROM_PHONE_NUMBER_ID}/messages'

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Content-Type": "application/json"
}

# data = {
#     "messaging_product": "whatsapp",
#     # "recipient_type": "individual",
#     "to": PHONE_NUMBER,
#     # "type": "text",
#     "text": {
#              "body": "MESSAGE_CONTENT"
#              }
# }

data = {
    "messaging_product": "whatsapp",
    "to": PHONE_NUMBER,
    "type": "template",
    "template": {
        "name": "templa",
        "language": {
            "code": "es"
        },
        "components": [
            {
               "type": "header",
               "parameters": [
                   {
                       "type": "text",
                       "text": "Hola Santy"
                   },
               ]
           },
           {
               "type": "body",
               "parameters": [
                   {
                       "type": "text",
                       "text": "*Mr. Jones*"
                   },
                   {
                       "type": "text",
                       "text": "*Mr. Jones*"
                   },
                   {
                       "type": "text",
                       "text": "*Mr. Jones*"
                   },
                   {
                       "type": "text",
                       "text": "*Mr. Jones*"
                   },


               ]
           }
        ]
    }
}

# data = {
#     "messaging_product": "whatsapp",
#     "to": PHONE_NUMBER,
#     "type": "template",
#     "template": {
#        "name": "sample_shipping_confirmation",
#        "language": {
#            "code": "en_US",
#            "policy": "deterministic"
#        },
#        "components": [
#          {
#            "type": "body",
#            "parameters": [
#                {
#                    "type": "text",
#                    "text": "2"
#                }
#            ]
#          }
#        ]
#     }
# }

print(requests.post(endpoint, headers=headers, json=data).json())
