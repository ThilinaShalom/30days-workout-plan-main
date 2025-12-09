'use client';
import Link from 'next/link';
import { Navbar, Container, Nav, Button } from 'react-bootstrap';
import { useRouter } from 'next/navigation';
import api from '@/lib/api';

interface AppNavbarProps {
    brand: string;
}

export default function AppNavbar({ brand }: AppNavbarProps) {
    const router = useRouter();

    const handleLogout = async () => {
        try {
            await api.post('/auth/logout');
            // Clear any local storage if needed
            localStorage.clear();
            sessionStorage.clear();
            router.push('/login');
        } catch (error) {
            console.error('Logout failed', error);
            // Even if API call fails, redirect to login
            router.push('/login');
        }
    };

    return (
        <Navbar expand="lg" variant="dark" style={{ background: 'linear-gradient(90deg, #0077ff, #00ccff)', boxShadow: '0 0 10px rgba(0, 225, 255, 0.6)' }}>
            <Container>
                <Navbar.Brand href="#">{brand}</Navbar.Brand>
                <Navbar.Toggle aria-controls="navbarNav" />
                <Navbar.Collapse id="navbarNav">
                    <Nav className="ms-auto">
                        <Nav.Link onClick={handleLogout} style={{ cursor: 'pointer', color: 'white', fontWeight: 600 }}>Logout</Nav.Link>
                    </Nav>
                </Navbar.Collapse>
            </Container>
        </Navbar>
    );
}
