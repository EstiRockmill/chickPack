using EasyPack.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Core.Repositories
{
    public interface IUserRepository
    {
        List<User> GetUserList();
        User GetUserById(int id);
        User GetUserByEmail(string email);
        void DeleteUser(int id);
        User UpdateUser(User user, int userId);
        User AddUser(User user);
    }
}
