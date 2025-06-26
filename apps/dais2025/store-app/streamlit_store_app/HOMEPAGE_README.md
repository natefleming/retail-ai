# Modular Card-Based Homepage Layout

## Overview

This implementation provides a comprehensive, role-based homepage layout for the retail store employee iPad app. The layout dynamically adapts based on the user's assigned role and utilizes modular card components for optimal user experience.

## Key Features

### 🎯 Role-Based Content
- **Store Manager View**: KPIs, operational briefs, team insights
- **Store Associate View**: Assigned tasks, personal schedule, product highlights
- **Universal Components**: Notifications and inventory status for all roles

### 🧩 Modular Components
- **Metric Cards**: Display key performance indicators and inventory status
- **Alert Widgets**: Show notifications and system alerts
- **Task Cards**: Organize BOPIS orders, restocking, and other assignments
- **Schedule Cards**: Personal and team scheduling information
- **Promotion Cards**: Product highlights and current promotions

### 💬 Persistent Chat Widget
- Always accessible AI assistant
- Floating button with expandable interface
- Integrated with existing chat functionality

## Component Architecture

### Store Manager Components

#### 1. KPI Dashboard (`show_kpi_dashboard()`)
- **Today's Sales**: Revenue with day-over-day comparison
- **Order Status**: Completion rates and pending orders
- **Staff Status**: Current staffing levels vs. scheduled
- **Inventory Alerts**: Critical stock notifications

#### 2. Daily Operations Brief (`show_daily_operations_brief()`)
- **Today's Priorities**: Task list with status indicators
- **Quick Stats**: Customer traffic, transaction averages, peak hours

#### 3. Team Insights (`show_team_insights()`)
- **Top Performers**: Employee performance metrics
- **Schedule Overview**: Shift coverage and gaps

### Store Associate Components

#### 1. Assigned Tasks (`show_assigned_tasks()`)
- **BOPIS Orders**: Customer pickup orders with priorities
- **Restocking Tasks**: Inventory replenishment assignments
- **Other Tasks**: Customer assistance and miscellaneous duties

#### 2. Personal Schedule (`show_personal_schedule()`)
- **Today's Shift**: Current shift details and break times
- **Upcoming Shifts**: Future schedule preview
- **Weekly Stats**: Hours worked and performance metrics

#### 3. Product & Promotions (`show_product_promotions()`)
- **Today's Promotions**: Active sales and discounts
- **Featured Products**: Important items and stock levels

### Universal Components

#### 1. Notifications (`show_notifications()`)
- System alerts and announcements
- Categorized by importance (info, warning, error)
- Timestamped for relevance

#### 2. Inventory Status (`show_inventory_status()`)
- **Critical Stock**: Items requiring immediate attention
- **Low Stock**: Items below threshold
- **Well Stocked**: Items in good supply
- **Overstock**: Items above capacity

#### 3. Persistent Chat (`show_persistent_chat()`)
- Floating chat button (bottom-right corner)
- Expandable interface for AI assistance
- Maintains chat history and context

## Technical Implementation

### File Structure
```
streamlit_store_app/
├── components/
│   ├── homepage.py          # All homepage components
│   ├── styles.py           # Enhanced CSS styles
│   ├── metrics.py          # Metric card components
│   └── __init__.py         # Component exports
├── app.py                  # Updated main application
└── utils/
    └── store_context.py    # Role and store management
```

### Key Functions

#### Main Layout Functions
- `show_home()`: Main homepage coordinator
- `show_manager_homepage()`: Manager-specific layout
- `show_associate_homepage()`: Associate-specific layout

#### Component Functions
- `show_kpi_dashboard()`: Manager KPI metrics
- `show_assigned_tasks()`: Associate task management
- `show_notifications()`: Universal alert system
- `show_inventory_status()`: Universal inventory overview
- `show_persistent_chat()`: AI assistant interface

### CSS Classes

#### Card Components
- `.kpi-card`: KPI dashboard cards
- `.task-card`: Task assignment cards
- `.schedule-card`: Schedule information cards
- `.promotion-card`: Promotion highlight cards
- `.inventory-metric-card`: Inventory status cards

#### Interactive Elements
- `.task-action`: Task action buttons
- `.chat-widget-container`: Floating chat button
- `.priority-item`: Priority task items

## Usage

### Running the Application
```bash
cd streamlit_store_app
streamlit run app.py
```

### Testing Components
```bash
streamlit run test_homepage.py
```

### Role Selection
1. Use the sidebar to select a store and role
2. Homepage content automatically adapts
3. Common components remain accessible to all roles

## Responsive Design

The layout is optimized for iPad use with:
- **Touch-friendly interfaces**: Large buttons and cards
- **Responsive grid layouts**: Adapts to different screen orientations
- **Clear visual hierarchy**: Easy scanning and navigation
- **Consistent spacing**: Comfortable viewing distances

## Data Integration

### Mock Data
Currently uses mock data for demonstration. Replace with actual database queries:
- Sales metrics from POS systems
- Task assignments from workforce management
- Inventory levels from stock management systems
- Schedule data from HR systems

### Database Integration Points
- `query()` function calls in components
- Store context from `st.session_state.store_id`
- User permissions from `check_permission()`

## Customization

### Adding New Components
1. Create component function in `components/homepage.py`
2. Add CSS styles in `components/styles.py`
3. Import and use in appropriate homepage layout
4. Update `__init__.py` exports

### Modifying Layouts
- Adjust column ratios in layout functions
- Reorder components based on priority
- Add conditional rendering based on permissions

### Styling Updates
- Modify CSS classes in `styles.py`
- Update color schemes and spacing
- Add animations and transitions

## Performance Considerations

- **Lazy Loading**: Components only render when needed
- **Efficient Queries**: Minimize database calls
- **Caching**: Use Streamlit caching for static data
- **Responsive Images**: Optimize for mobile bandwidth

## Future Enhancements

1. **Real-time Updates**: WebSocket integration for live data
2. **Customizable Dashboards**: User-configurable layouts
3. **Advanced Analytics**: Deeper performance insights
4. **Mobile Optimization**: Enhanced touch interactions
5. **Offline Support**: Local data caching capabilities 