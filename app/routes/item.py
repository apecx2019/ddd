from fastapi import APIRouter

router = APIRouter()

fake_items = [
    {"id": 1, "name": "Keyboard", "price": 500},
    {"id": 2, "name": "Mouse", "price": 300},
]

@router.get("/")
def get_items():
    return fake_items

# @router.post("/")
# def create_item(item: Item):
#     new_item = {"id": len(fake_items) + 1, **item.dict()}
#     fake_items.append(new_item)
#     return new_item
