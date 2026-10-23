from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent import _conversation_context
from app.models import Message
from app.train_emotion_model import _go_emotion_label, _text_label


def test_text_dataset_label_mapping():
    assert _text_label({"label": "0"}) == "sadness"
    assert _text_label({"label": "5"}) == "surprise"
    assert _text_label({"label": "unknown"}) is None


def test_go_emotions_uses_only_clear_single_group_rows():
    row = {
        "example_very_unclear": "0",
        "joy": "1",
        "sadness": "0",
        "love": "0",
    }
    assert _go_emotion_label(row) == "joy"
    assert _go_emotion_label({**row, "sadness": "1"}) is None
    assert _go_emotion_label({**row, "example_very_unclear": "1"}) is None


def test_chat_context_is_recent_ordered_and_user_scoped():
    engine = create_engine("sqlite:///:memory:")
    Message.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    db.add_all([
        Message(user_id=1, timestamp="1", role="user", content="Hello"),
        Message(user_id=1, timestamp="2", role="assistant", content="Hi there"),
        Message(user_id=2, timestamp="3", role="user", content="Other user"),
        Message(user_id=1, timestamp="4", role="tool", content="tool data"),
        Message(user_id=1, timestamp="5", role="user", content="Follow-up"),
    ])
    db.commit()

    assert _conversation_context(db, 1) == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "Follow-up"},
    ]

    db.close()
    engine.dispose()
