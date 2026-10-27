from app.database import load_orders,load_faqs

faqs = load_faqs()
orders = load_orders()

print(f"Total FAQs: {len(faqs)}")
print(f"Total orders: {len(orders)}")

print("\nFirst FAQ:")
print(faqs[0])

print("\nFirst order:")
print(orders[0])