import os
import csv
from abc import ABC, abstractmethod
from contact_plan.contact_plan import ContactPlan, Contact


class ContactPlanParser(ABC):
    SOURCES_ROOT = "experiments"
    OUTPUT_SUFFIX = "scheduled"

    @abstractmethod
    def read(self, file_name: str) -> ContactPlan:
        pass

    @abstractmethod
    def write(self, file_name: str, contact_plan: ContactPlan):
        pass


class IONContactPlanParser(ContactPlanParser):
    
    CONTACT_PREFIX = ["a", "contact"]
    TIMESTAMP_PREFIX = "+"
    BIT_RATE_CONTEXT = "bit_rate"
    
    def read(self, file_name: str) -> ContactPlan:
        contacts = []

        path = os.path.join(ContactPlanParser.SOURCES_ROOT, file_name)
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                ion_start_time, ion_end_time, tx_node, rx_node, bit_rate = row[2:]

                contact = Contact(
                    tx_node=tx_node,
                    rx_node=rx_node,
                    start_time=int(ion_start_time[1:]),
                    end_time=int(ion_end_time[1:]),
                    context={
                        IONContactPlanParser.BIT_RATE_CONTEXT: float(bit_rate)
                    }
                )

                contacts.append(contact)

        return ContactPlan(contacts)

    def write(self, file_name: str, contact_plan: ContactPlan):
        path = os.path.join(ContactPlanParser.SOURCES_ROOT, f"{file_name}_{ContactPlanParser.OUTPUT_SUFFIX}")
        
        rows = []
        for contact in contact_plan.contacts:
            ion_start_time = f"{IONContactPlanParser.TIMESTAMP_PREFIX}{contact.start_time}"
            ion_end_time = f"{IONContactPlanParser.TIMESTAMP_PREFIX}{contact.end_time}"

            row = [
                ion_start_time,
                ion_end_time,
                contact.tx_node,
                contact.rx_node,
                contact.context[IONContactPlanParser.BIT_RATE_CONTEXT]
            ]
            
            rows.append(IONContactPlanParser.CONTACT_PREFIX + row)

        with open(path, "w") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerows(rows)
