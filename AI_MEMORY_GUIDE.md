# How to Ensure AI Assistants Remember Lessons

## Quick Answer

**Use the memory system** - Ask the AI to create a memory entry for critical lessons. This persists across conversations.

## Current Setup

We've already created multiple layers of documentation:

1. **DEVELOPMENT_GUIDELINES.md** - Comprehensive guide
2. **Code comments** - Contextual reminders
3. **Test docstrings** - Reminders when running tests
4. **README.md** - Easy to find
5. **.cursorrules_lessons** - Quick reference

## For AI Assistants

### Memory System (Best Method)
Ask the AI: "Create a memory entry for [lesson]"
- Persists across conversations
- Automatically retrieved when relevant
- Can be updated if needed

### Documentation (Secondary)
- Well-structured docs are easy to search
- AI can find and reference them
- But requires explicit search/reading

### Code Comments (Tertiary)
- Contextual reminders in relevant code
- AI sees them when reading code
- But only when working on that code

## Example Memory Entry

When you ask me to remember something, I'll create a memory like:

```
Memory: "For AV stack control logic, always test empirically 
rather than reasoning theoretically. The steering direction 
issue (2025-12-25) showed that theoretical reasoning about 
sign conventions can be wrong. Always write tests first for 
control logic, especially for direction/correction behavior. 
See DEVELOPMENT_GUIDELINES.md for full details."
```

## Best Practices

1. **Create memory for critical lessons** - Things that caused bugs
2. **Reference documentation** - Point to where full details are
3. **Be specific** - Include what happened and how to prevent it
4. **Update if needed** - If the lesson changes, update the memory

## Current Memory Entries

- Steering direction lesson (test empirically, not theoretically)
- See DEVELOPMENT_GUIDELINES.md for full context

## How to Use

Just ask: "Make sure you remember [lesson]" and I'll create a memory entry!
