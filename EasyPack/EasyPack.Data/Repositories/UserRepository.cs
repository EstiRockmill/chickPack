using EasyPack.Core.Models;
using EasyPack.Core.Repositories;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Data.Repositories
{
    public class UserRepository : IUserRepository
    {
        private readonly DataContext _context;

        public UserRepository(DataContext context)
        {
            _context = context;
        }
        public User AddUser(User user)
        {
            _context.Users.Add(user);
            _context.SaveChanges();
            return user;
        }

        public void DeleteUser(int id)
        {
            User b = GetUserById(id);
            _context.Users.Remove(b);
            _context.SaveChanges();
        }

        public User GetUserById(int id)
        {
            return _context.Users.FirstOrDefault(b => b.Id == id);
        }
        public User GetUserByEmail(string email)
        {
            return _context.Users.FirstOrDefault(b => EF.Functions.Like(b.Email, email));
        }

        public List<User> GetUserList()
        {
            return _context.Users.Include(u=>u.Rooms).ToList();
        }

        public User UpdateUser(User user, int UserId)
        {
            User u = GetUserById(UserId);
            u.Rooms = user.Rooms;
            u.Name = user.Name;
            u.Address = user.Address;
            u.Email = user.Email;
            u.Password = user.Password;
            _context.Users.Update(u);
            _context.SaveChanges();
            return u;
        }
    }
}
