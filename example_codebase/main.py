"""
Example Python application for testing the AI analyzer.
"""

class UserManager:
    def __init__(self):
        self.users = {}
    
    def create_user(self, username, email):
        """Create a new user account."""
        if username in self.users:
            raise ValueError("User already exists")
        
        self.users[username] = {
            'email': email,
            'active': True,
            'created_at': datetime.now()
        }
        return True
    
    def authenticate_user(self, username, password):
        """Authenticate user login."""
        user = self.users.get(username)
        if not user or not user.get('active'):
            return False
        
        # In real app, check password hash
        return True

def main():
    """Main application entry point."""
    manager = UserManager()
    
    # Example usage
    manager.create_user("john_doe", "john@example.com")
    
    if manager.authenticate_user("john_doe", "password"):
        print("Login successful!")
    else:
        print("Login failed!")

if __name__ == "__main__":
    main()
