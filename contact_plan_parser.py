from abc import ABC, abstractmethod


class ContactPlanParser(ABC):

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def write(self):
        pass


class IONContactPlanParser(ContactPlanParser):

    def read(self):
        pass

    def write(self):
        pass
