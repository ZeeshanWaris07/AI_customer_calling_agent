
from app.database import get_order_by_id


order_id = "ORD-1001"

order = get_order_by_id(order_id)

if order:
    print("Order found!")
    print("Order ID:", order["order_id"])
    print("Customer ID:", order["customer_id"])
    print("Status:", order["status"])
    print("Estimated delivery:", order["estimated_delivery"])

else:
    print("Order not found.")


print("\nTesting nonexistent order:")

missing_order = get_order_by_id("ORD-9999")

print("Result:", missing_order)
