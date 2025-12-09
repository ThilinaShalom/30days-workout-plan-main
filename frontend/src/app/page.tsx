'use client';
import { useEffect } from 'react';
import Navbar from '@/components/Navbar';
import Hero from '@/components/Hero';
import Features from '@/components/Features';
import MLProcess from '@/components/MLProcess';
import Stats from '@/components/Stats';
import Footer from '@/components/Footer';

export default function Home() {
  useEffect(() => {
    // Cursor Glow Effect
    const cursor = document.createElement('div');
    cursor.className = 'cursor-glow';
    document.body.appendChild(cursor);

    const moveCursor = (e: MouseEvent) => {
      cursor.style.left = e.clientX + 'px';
      cursor.style.top = e.clientY + 'px';
    };

    window.addEventListener('mousemove', moveCursor);

    // Gym Equipment Animation
    const createParticle = (x: number, y: number) => {
      const particle = document.createElement('div');
      particle.className = 'particle';
      const size = Math.random() * 5 + 2;
      particle.style.width = size + 'px';
      particle.style.height = size + 'px';
      particle.style.left = x + 'px';
      particle.style.top = y + 'px';

      const destinationX = (Math.random() - 0.5) * 100;
      const destinationY = (Math.random() - 0.5) * 100;

      particle.animate([
        { transform: 'translate(0, 0)', opacity: 1 },
        { transform: `translate(${destinationX}px, ${destinationY}px)`, opacity: 0 }
      ], {
        duration: 800,
        easing: 'ease-out'
      }).onfinish = () => particle.remove();

      document.body.appendChild(particle);
    };

    const equipmentEmojis = ['🏋️', '💪', '🏃', '🧘', '🚴'];
    const dropEquipment = () => {
      if (Math.random() > 0.05) return; // Low chance

      const equipment = document.createElement('div');
      equipment.className = 'gym-equipment';
      equipment.textContent = equipmentEmojis[Math.floor(Math.random() * equipmentEmojis.length)];
      equipment.style.left = Math.random() * window.innerWidth + 'px';
      equipment.style.top = '-50px';

      document.body.appendChild(equipment);

      setTimeout(() => {
        equipment.remove();
      }, 1000);
    };

    const handleMouseMove = (e: MouseEvent) => {
      // Create particles
      for (let i = 0; i < 3; i++) {
        createParticle(e.clientX, e.clientY);
      }
      dropEquipment();
    };

    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      window.removeEventListener('mousemove', moveCursor);
      window.removeEventListener('mousemove', handleMouseMove);
      if (cursor.parentNode) cursor.parentNode.removeChild(cursor);
    };
  }, []);

  return (
    <main>
      <Navbar />
      <Hero />
      <Features />
      <MLProcess />
      <Stats />
      <Footer />
    </main>
  );
}
