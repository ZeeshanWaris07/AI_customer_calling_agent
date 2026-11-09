
from app.database import seed_orders


count = seed_orders()

print(f"Orders processed: {count}")
print("Order migration completed.")
