/**
 * Simple REST API example
 */

const express = require('express');
const app = express();

// Middleware
app.use(express.json());

// In-memory database simulation
let users = [];
let nextId = 1;

// Routes
app.get('/api/users', (req, res) => {
    res.json(users);
});

app.post('/api/users', (req, res) => {
    const { name, email } = req.body;
    
    if (!name || !email) {
        return res.status(400).json({ error: 'Name and email required' });
    }
    
    const user = {
        id: nextId++,
        name,
        email,
        createdAt: new Date()
    };
    
    users.push(user);
    res.status(201).json(user);
});

app.get('/api/users/:id', (req, res) => {
    const user = users.find(u => u.id === parseInt(req.params.id));
    
    if (!user) {
        return res.status(404).json({ error: 'User not found' });
    }
    
    res.json(user);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
