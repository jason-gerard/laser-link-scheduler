import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from utils import get_experiment_file, FileType


@dataclass
class Contact:
    # Transmitting node, from node
    tx_node: str
    # Receiving node, to node
    rx_node: str
    start_time: int
    end_time: int
    # The context attribute can be any additional information associated with the contact. For example this could
    # include the range or distance between the contacts, one way light time (OWLT), data rate, BER, etc
    context: dict


@dataclass
class ContactPlan:
    contacts: list[Contact]


class ContactPlanParser(ABC):
    @abstractmethod
    def read(self, file_name: str) -> ContactPlan:
        pass

    @abstractmethod
    def write(self, file_name: str, contact_plan: ContactPlan, file_type: FileType):
        pass


class IONContactPlanParser(ContactPlanParser):
    
    CONTACT_PREFIX = ["a", "contact"]
    TIMESTAMP_PREFIX = "+"
    DURATION_CONTEXT = "duration"
    RANGE_CONTEXT = "range"

    def read(self, experiment_name: str) -> ContactPlan:
        contacts = []

        path = get_experiment_file(experiment_name, FileType.CONTACT_PLAN)
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                if not row:
                    continue
                
                ion_start_time, ion_end_time, tx_node, rx_node, duration, contact_range = row[2:]

                contact = Contact(
                    tx_node=tx_node,
                    rx_node=rx_node,
                    start_time=int(ion_start_time[1:]),
                    end_time=int(ion_end_time[1:]),
                    context={
                        IONContactPlanParser.DURATION_CONTEXT: float(duration),
                        IONContactPlanParser.RANGE_CONTEXT: float(contact_range)
                    }
                )

                contacts.append(contact)

        return ContactPlan(contacts)

    def write(self, experiment_name: str, contact_plan: ContactPlan, file_type: FileType):
        rows = []

        path = get_experiment_file(experiment_name, file_type)
        for contact in contact_plan.contacts:
            ion_start_time = f"{IONContactPlanParser.TIMESTAMP_PREFIX}{contact.start_time}"
            ion_end_time = f"{IONContactPlanParser.TIMESTAMP_PREFIX}{contact.end_time}"

            row = [
                ion_start_time,
                ion_end_time,
                contact.tx_node,
                contact.rx_node,
            ]
            
            # Add optional context values to the row
            if IONContactPlanParser.DURATION_CONTEXT in contact.context:
                rows.append(contact.context[IONContactPlanParser.DURATION_CONTEXT])
            if IONContactPlanParser.RANGE_CONTEXT in contact.context:
                rows.append(contact.context[IONContactPlanParser.RANGE_CONTEXT])
            
            rows.append(IONContactPlanParser.CONTACT_PREFIX + row)

        with open(path, "w") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerows(rows)


def contact_plan_splitter(contact_plan: ContactPlan) -> ContactPlan:
    return contact_plan
