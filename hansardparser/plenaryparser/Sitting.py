"""Defines the Sitting class used in hansard_parser.py.

Each Sitting object represents a single sitting of the National Assembly, containing metadata such as time and date of the sitting. 

NOTE TO SELF: not clear whether each transcript actually counts as a sitting. Will need to write a method that checks whether transcript is a new sitting or a continuation.

NOTE TO SELF: come back to this later and make it such that date and time are a single attribute (containing a datetime object).
"""

class Sitting(object):
    """ Meta-data from a Hansard transcript.
    
    Attributes:
        heading: str
            title of the transcript
        date: date obj
            Date of the sitting.
        time: str
            Time of the sitting
        start_page: int
            starting page of the transcript.
        sitting : int
            sitting of parliament.
        session : int
            session of parliament
        parliament : int
            e.g. 7th parliament. 
    """

    # __metaclass__ = ABCMeta

    def __init__(self, heading=None, date=None, time=None, start_page=None, sitting=None, session=None, parliament=None):
        self.heading = heading
        self.date = date
        self.time = time
        self.start_page = start_page
        self.sitting = sitting
        self.session = session
        self.parliament = parliament

    def __str__(self):
        return str(self.__dict__)

    def is_incomplete(self):
        return self.heading is None or self.date is None or self.time is None or self.start_page is None
        # 'Heading: ' + str(self.heading) + '\nDate: ' + str(self.date) + '\nTime: ' +  str(self.time) + '\nStart page: ' +  str(self.start_page)
