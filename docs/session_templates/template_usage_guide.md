# How to Use the Chat Context Template

## Purpose

This template allows you to start fresh chat sessions while maintaining full project context, helping manage API usage limits while ensuring continuity of development.

## Files to Keep Updated

### 1. **PROJECT_TRACKER.md** (Primary)

- Update after each work session
- Mark completed tasks with ✅
- Update the "Current Status" section
- Add notes to the Worklog
- Update test results if new tests are run

### 2. **Module Mapping Guide** (Reference)

- Generally static, update only if architecture changes
- Shows how modules map to milestones

### 3. **settings.yaml** (If Changed)

- Update in template if you modify configuration
- Especially important for risk parameters or data sources

## Step-by-Step Usage

### Before Starting a New Chat:

1. **Update PROJECT_TRACKER.md**

   - Mark any completed items
   - Update current status
   - Note what needs to be done next

2. **Copy the Template**

   - Open `chat_context_template.md`
   - Copy the entire contents

3. **Insert Current Files**

   - Replace `[PASTE CURRENT PROJECT_TRACKER.md HERE]` with actual content
   - Replace `[PASTE MODULE MAPPING GUIDE HERE]` with actual content
   - Update the "Next Task" section based on tracker

4. **Add Session-Specific Context** (Optional)
   - Any errors from last session
   - Specific decisions that need to be continued
   - Test results that need investigation

### Starting the New Chat:

1. **Paste the Complete Template**

   - Include all sections with actual content inserted

2. **Verify Understanding**

   - Ask: "Can you confirm you understand the project context and what needs to be done next?"
   - The assistant should identify the current milestone and next modules to implement

3. **Continue Development**
   - The assistant will pick up exactly where you left off
   - All architectural decisions and patterns will be maintained

## What to Include vs Exclude

### Always Include:

- ✅ PROJECT_TRACKER.md (current version)
- ✅ Module Mapping Guide
- ✅ Current milestone and next tasks
- ✅ Working CLI commands
- ✅ Key test results to verify

### Include if Changed:

- ⚠️ settings.yaml (if modified)
- ⚠️ New architectural decisions
- ⚠️ Error messages or blockers
- ⚠️ New test results

### Don't Include (unless relevant):

- ❌ Full code implementations (assistant can recreate)
- ❌ Detailed test outputs (just summaries)
- ❌ Historical discussions (just decisions made)

## Example First Message in New Chat

```markdown
[Paste the complete filled template]

Please confirm you understand the project context and identify what module we should implement next based on the PROJECT_TRACKER.md.
```

## Maintaining Continuity

### After Each Session:

1. Update PROJECT_TRACKER.md with progress
2. Note any new decisions in "Key Decisions Made"
3. Update architecture status if new modules completed
4. Add to worklog with timestamp

### Version Control:

Consider keeping versions of:

- PROJECT_TRACKER_YYYY-MM-DD.md after major milestones
- Full implementation artifacts (save locally)
- Test results that validate functionality

## Quick Checklist for New Session

- [ ] PROJECT_TRACKER.md is current
- [ ] "Next Task" section is clear
- [ ] Test results are noted
- [ ] Any blockers are documented
- [ ] Template is fully filled out
- [ ] Previous session's code is saved locally

## Benefits of This Approach

1. **Cost Management**: Fresh sessions use less context
2. **Clean State**: No accumulated conversation overhead
3. **Consistency**: Same context every time
4. **Progress Tracking**: Clear record of advancement
5. **Easy Handoff**: Could even hand off to another person

## Troubleshooting

If the assistant seems confused:

1. Ensure PROJECT_TRACKER.md is complete
2. Verify the "Next Task" section is specific
3. Include any error messages from last session
4. Ask for confirmation of understanding before proceeding

## Pro Tips

1. **Save Artifacts**: Download code artifacts after each session
2. **Git Commits**: Commit after each successful milestone
3. **Test First**: Run tests to verify previous work before continuing
4. **Small Sessions**: Complete one module per session for clarity
5. **Document Decisions**: Add important decisions to tracker immediately

---

This template system ensures seamless continuation across sessions while optimizing token usage and maintaining full project context.
