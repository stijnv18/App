from homeassistant_api import Client,State
import time
homeass = "http://192.168.69.130:8123/api/"
homeass_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJlYjA1MTZiMDliMDE0NjI5YTViZWVhZGFlMmQ3ODcwNiIsImlhdCI6MTcxODI4NDUzOSwiZXhwIjoyMDMzNjQ0NTM5fQ.c36_A67ImCeG9pecsC7Ec7pYCQkj7omKwgQvQyEKU7M"
# Initialize Home Assistant client

haclient = Client(homeass, homeass_token)

# Get the states
states = haclient.get_states()



# Function to switch on the light
def switch_on_light():
    print("Switching on the light")    
    #new_state = haclient.set_state(State(entity_id='switch.shellyplusplugs_e86beae87700_switch_0', state='on'))
    haclient.trigger_service("switch", "turn_off", entity_id="switch.shellyplusplugs_e86beae87700_switch_0")
    print("Light switched on successfully")

# Call the function to switch on the light
switch_on_light()
