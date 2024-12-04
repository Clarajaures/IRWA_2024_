import json
import os

class AnalyticsData:
    """
    An in-memory persistence object.
    Declare more variables to hold analytics tables.
    """
    # statistics table 1
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter
    fact_clicks = {}

    # statistics table 2 --> id
    fact_two = {}

    # statistics table 3 --> agent
    fact_three = {}

    # statistics table 4 --> query terms
    fact_query_terms = {}

    def __init__(self):
        # Initialize analytics data when the object is created
        self.open_data()  # Load existing data from the JSON file at startup

    def open_data(self):
        """
        Open and load data from the JSON file if it exists.
        """
        if os.path.exists('data_collection.json'):
            print("Opening data_collection.json...")

            # Open the existing data file
            with open('data_collection.json', 'r') as f:
                data = json.load(f)

                # Populate internal dictionaries with loaded data or empty dictionaries if not found
                self.fact_clicks = data.get('fact_clicks', {})
                self.fact_two = data.get('fact_two', {})
                self.fact_three = data.get('fact_three', {})
                self.fact_query_terms = data.get('fact_query_terms', {})

            print("Data loaded successfully.")
        else:
            # If the file does not exist, create it and save initial empty data
            print("data_collection.json not found. Creating new file...")
            self.save_data()

    def save_data(self):
        """
        Save the current state of analytics data to the JSON file.
        """
        print("Saving data to data_collection.json...")

        # Write the current state of the analytics data to the file
        with open('data_collection.json', 'w') as f:
            json.dump({
                'fact_clicks': self.fact_clicks,
                'fact_two': self.fact_two,
                'fact_three': self.fact_three,
                'fact_query_terms': self.fact_query_terms
            }, f)

        print("Data saved successfully.")

    def save_query_terms(self, terms: str) -> int:
        """
        Increment the count for a specific query term in the analytics data.
        """
        if terms in self.fact_query_terms:
            self.fact_query_terms[terms] += 1
        else:
            self.fact_query_terms[terms] = 1

        self.save_data()  # Save data after updating query terms
        return 0

    def save_clicks(self):
        """
        Save click data.
        """
        self.save_data()  # Save data after updating click data
        return 0

    def save_session_terms(self, ip, agent) -> int:
        """
        Save session-related data like IP address and user agent.
        """
        # Update the click counter for the IP address
        if ip in self.fact_two:
            self.fact_two[ip] += 1
        else:
            self.fact_two[ip] = 1

        # Convert agent info to string format
        agent_ = str(agent)
        
        # Save user agent details
        if agent_ in self.fact_three:
            self.fact_three[agent_] = [
                1 + self.fact_three[agent_][0], agent["platform"]["name"], 
                agent["bot"], agent["browser"]["name"], agent["browser"]["version"]
            ]
        else:
            self.fact_three[agent_] = [
                1, agent["platform"]["name"], agent["bot"],
                agent["browser"]["name"], agent["browser"]["version"]
            ]

        self.save_data()  # Save data after updating session data
        return 0


class ClickedDoc:
    """
    Class to represent a clicked document and its analytics data.
    """
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        """
        Convert the object to a JSON-compatible dictionary.
        """
        return self.__dict__
    
    def to_dict(self):
        """Convert the ClickedDoc instance into a dictionary that can be serialized into JSON."""
        return {
            "doc_id": self.doc_id,
            "description": self.description,
            "counter": self.counter
        }

    def __str__(self):
        """
        Return a string representation of the object in JSON format.
        """
        return json.dumps(self)
