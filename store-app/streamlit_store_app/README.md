# Store Companion App

A Streamlit application for retail store management that integrates with Databricks SQL warehouse.

## Features

- **Dynamic Store Loading**: Automatically loads store data from `demos_genie.rcg_store_manager_gold.dim_stores` table
- **Employee Management**: Configurable employee names for demo purposes
- **Role-based Access**: Store Associate and Store Manager roles with different permissions
- **Real-time Data**: Direct integration with Databricks SQL warehouse

## Quick Start

### 1. Environment Setup

**For Contributors**: Copy the example environment file and configure your settings:

```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your actual Databricks credentials
# (See Configuration section below for details)
```

**Required Environment Variables:**
- `DATABRICKS_HOST` - Your Databricks workspace hostname
- `DATABRICKS_TOKEN` - Personal access token or service principal token  
- `DATABRICKS_WAREHOUSE_ID` - SQL warehouse ID for database connections
- `SERVING_ENDPOINT` - Name of your deployed agent serving endpoint

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

## Configuration

### Environment Variables

The app uses environment variables for configuration. **Use the `env.example` file as a template:**

```bash
# Copy and customize the example file
cp env.example .env
```

**Required variables:**
```bash
DATABRICKS_HOST=your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DATABRICKS_WAREHOUSE_ID=abc123def456
SERVING_ENDPOINT=retail_multi_agent
```

**Optional variables:**
```bash
DEBUG=true                    # Enable debug mode
USE_MOCK_DATA=false          # Use real database vs mock data
DATABASE_TIMEOUT=30          # Connection timeout in seconds
```

See `env.example` for a complete list of all available configuration options.

### Database Connection

The app connects to Databricks SQL warehouse using the environment variables above.

For testing without database access, set `USE_MOCK_DATA=true` in your `.env` file.

### Store Data Source

Set `database.mock: false` in `config.yaml` to use real Databricks data:

```yaml
database:
  mock: false  # Use real Databricks database
  connection_timeout: 30
```

### Employee Configuration

Configure employee names for demo purposes in `config.yaml`:

```yaml
employees:
  store_manager:
    name: "Sarah Johnson"
  store_associate:
    name: "Victoria Chen"
```

## Database Schema

The app expects the following table structure:

```sql
demos_genie.rcg_store_manager_gold.dim_stores
```

With columns:
- `store_id`: Unique store identifier
- `store_name`: Display name of the store
- `store_address`: Street address
- `store_city`: City name
- `store_state`: State abbreviation
- `store_zipcode`: ZIP code
- `store_phone`: Contact phone number
- `store_email`: Contact email
- `store_area_sqft`: Store size in square feet
- `is_open_24_hours`: Boolean for 24/7 operation
- `latitude`, `longitude`: Geographic coordinates
- `region_id`: Regional identifier

## Usage

1. **Start the app**: `streamlit run app.py`
2. **Select Store**: Choose from dynamically loaded stores
3. **Select Role**: Pick Store Manager or Store Associate
4. **Welcome Message**: See personalized greeting with employee name

## Debug Mode

Enable debug mode in `config.yaml` to see database connection status:

```yaml
app:
  debug: true
```

This will show:
- Database connection status
- Number of stores loaded
- Sample store data

## Fallback Mode

If database connection fails, the app automatically falls back to sample data to ensure functionality.

## Performance

- Store data is cached for 5 minutes to reduce database calls
- Fallback data is cached for 1 hour
- Automatic error handling and graceful degradation 